#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<cmath>
#include<cstdlib>
#include<sstream>
#include<algorithm>
using namespace std;

#define pi 3.1415926535897932384626433832795

map<pair<vector<int>,int>,double>  path_confidence;

bool L1_flag=false;
string version = "bern";
int REL_NUM = 6561;
int entity_space = 50;
int relation_space = 50;
double margin_test = 1;
double rate_test = 0.001;
string resultfile = "result_001_1_50_L2";
int hit_n = 10;
int rerank_num = 500;

char buf[100000];
int relation_num,entity_num;
map<string,int> relation2id,entity2id;
map<int,string> id2entity,id2relation;

map<int,double> left_mean,right_mean,relation_weight;
vector<int> relation_type;
vector<vector<pair<int,int> > > path;
map<pair<int,int>, map<int,int> > ok;

//normal distribution
double rand(double min, double max)
{
    return min+(max-min)*rand()/(RAND_MAX+1.0);
}
double normal(double x, double miu,double sigma)
{
    return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}
double randn(double miu,double sigma, double min ,double max)
{
    double x,y,dScope;
    do
    {
        x=rand(min,max);
        y=normal(x,miu,sigma);
        dScope=rand(0.0,normal(miu,miu,sigma));
    }
    while(dScope>y);
    return x;
}

double sqr(double x)
{
    return x*x;
}

double vec_len(vector<double> &a)
{
    double res=0;
    if (L1_flag)
        for (int i=0; i<a.size(); i++)
            res+=fabs(a[i]);
    else
    {
        for (int i=0; i<a.size(); i++)
            res+=a[i]*a[i];
        res = sqrt(res);
    }
    return res;
}

double cmp(pair<int,double> a, pair<int,double> b)
{
    return a.second>b.second;
}

class Test
{

public:
    void add(int x,int y,int z, vector<pair<vector<int>,double> > path_list)
    {
        if(z != -1)
        {
            fb_h.push_back(x);
            fb_r.push_back(z);
            fb_l.push_back(y);
        }
        fb_path[make_pair(x,y)] = path_list;
    }
    void run()
    {
        //n是实体空间维度
        n = entity_space;
        //m是关系空间维度
        m = relation_space;
        //学习率
        rate = rate_test;
        //边际
        margin = margin_test;
        cout<<"entity space:n="<<n<<' '<<"relation space:m="<<m<<endl;

        FILE* A_file = fopen((resultfile+"/A.txtbern").c_str(),"r");
        FILE* entity_vec_file = fopen((resultfile+"/entity2vec.txtbern").c_str(),"r");
        FILE* relation_vec_file = fopen((resultfile+"/relation2vec.txtbern").c_str(),"r");

        //A是将实体从实体空间映射到关系空间的投射矩阵
        A.resize(relation_num);
        //初始化A
        for (int i=0; i<relation_num; i++)
        {
            A[i].resize(m);
            for (int jj=0; jj<m; jj++)
            {
                A[i][jj].resize(n);
                for (int ii=0; ii<n; ii++)
                {
                    fscanf(A_file,"%lf",&A[i][jj][ii]);
                }
            }
        }

        relation_vec.resize(relation_num);
        for (int i=0; i<relation_vec.size(); i++)
        {
            relation_vec[i].resize(m);
            for(int j=0; j<m; j++)
            {
                fscanf(relation_vec_file,"%lf",&relation_vec[i][j]);
            }
        }

        entity_vec.resize(entity_num);
        for (int i=0; i<entity_vec.size(); i++)
        {
            entity_vec[i].resize(n);
            for(int j=0; j<n; j++)
            {
                fscanf(entity_vec_file,"%lf",&entity_vec[i][j]);
            }
        }

        fclose(A_file);
        fclose(entity_vec_file);
        fclose(relation_vec_file);

        entity_prediction();
        relation_predictive();
    }

private:
    int n,m;
    double margin;
    double res;
    double rate;//learning rate
    double belta;
    vector<int> fb_h,fb_l,fb_r;
    map<pair<int,int>,vector<pair<vector<int>,double> > >fb_path;
    vector<vector<double> > relation_vec,entity_vec;
    vector<vector<vector<double> > > A;

    void entity_prediction()
    {
        int fb_h_size = fb_h.size();
        cout<<"triplet nums: "<<fb_h_size<<endl;

        cout<<"start entity prediction!"<<endl;
		int l_raw_sum = 0,r_raw_sum = 0;
        int l_filter_sum = 0,r_filter_sum = 0;
        
        int l_raw_hit = 0,r_raw_hit = 0;
        int l_filter_hit = 0,r_filter_hit = 0;
		
        for(int testId = 0; testId<fb_h_size/2; testId++)
        {
            int e1 = fb_h[testId*2];
            int e2 = fb_l[testId*2];
            int rel = fb_r[testId*2];
            int rel_rev = rel + relation_num/2;

            vector<pair<int,double> > l_res,r_res;
            l_res.resize(entity_num);
            r_res.resize(entity_num);

            #pragma omp parallel for num_threads(240) schedule(dynamic,10)
            for(int entityId = 0; entityId<entity_num; entityId++)
            {
                //计算TransE
                double sum1 = 0,sum2 = 0;
                sum1+=calc_sum(entityId,e2,rel);
                sum1+=calc_sum(e2,entityId,rel_rev);
                l_res[entityId] = make_pair(entityId,sum1);
                sum2 += calc_sum(e1,entityId,rel);
                sum2 +=calc_sum(entityId,e1,rel_rev);
                r_res[entityId] = make_pair(entityId,sum2);
            }

            sort(l_res.begin(),l_res.end(),cmp);
            sort(r_res.begin(),r_res.end(),cmp);

            int l_raw_num = 0,r_raw_num = 0;
            int l_filter_num = 0,r_filter_num = 0;
            for(int i=0; i<entity_num; i++)
            {
                if(e1 != l_res[i].first)
                {
                    l_raw_num ++;
                    if(ok[make_pair(l_res[i].first,rel)].count(e2) == 0)
                    {
                        l_filter_num++;
                    }
                }
                else
                    break;
            }//for(int i=0; i<rerank_num; i++)
            for(int i=0; i<entity_num; i++)
            {
                if(e2 != r_res[i].first)
                {
                    r_raw_num++;
                    if(ok[make_pair(e1,rel)].count(r_res[i].first) == 0)
                    {
                        r_filter_num++;
                    }
                }
                else
                    break;
            }//for(int i=0;i<rerank_num;i++)

            cout<<l_raw_num<<"--"<<l_filter_num<<"--"<<r_raw_num<<"--"<<r_filter_num<<endl;

            l_raw_sum+=l_raw_num;
            l_filter_sum+=l_filter_num;
            r_raw_sum+=r_raw_num;
            r_filter_sum+=r_filter_num;
            if(l_raw_num+1<=10)
            {
            	l_raw_hit++;
            }
            if(l_filter_num+1<=10)
            {
            	l_filter_hit++;
            }
            if(r_raw_num+1<=10)
            {
            	r_raw_hit++;
            }
            if(r_filter_num+1<=10)
            {
            	r_filter_hit++;
            }
            /*if(testId%10 == 0)
                cout<<testId<<"--"<<fb_h_size/2<<endl;*/
        } //for(int testId = 0; testId<fb_h_size; testId++)
		cout<<"***mean rank***"<<endl;	
        cout<<"l_raw : "<<l_raw_sum/(fb_h_size/2)<<endl;
        cout<<"l_filter : "<<l_filter_sum/(fb_h_size/2)<<endl;
        cout<<"r_raw : "<<r_raw_sum/(fb_h_size/2)<<endl;
        cout<<"r_filter : "<<r_filter_sum/(fb_h_size/2)<<endl;
        cout<<"raw : "<<(l_raw_sum+r_raw_sum)/fb_h_size<<" filter : "<<(l_filter_sum+r_filter_sum)/fb_h_size<<endl;
        cout<<"***hits@10***"<<endl;
        cout<<"l_raw : "<<l_raw_hit/(fb_h_size/2)<<endl;
        cout<<"l_filter : "<<l_filter_hit/(fb_h_size/2)<<endl;
        cout<<"r_raw : "<<r_raw_hit/(fb_h_size/2)<<endl;
        cout<<"r_filter : "<<r_filter_hit/(fb_h_size/2)<<endl;
        cout<<"raw : "<<(l_raw_hit+r_raw_hit)/fb_h_size<<" filter : "<<(l_filter_hit+r_filter_hit)/fb_h_size<<endl;
        cout<<"entity prediction over!"<<endl;
    }//entity_prediction()

    void relation_predictive()
    {
        int fb_h_size = fb_h.size();

        cout<<"start relation prediction!"<<endl;
		int rel_raw_sum = 0,rel_filter_sum = 0;
		int rel_raw_hit = 0,rel_filter_hit = 0;
        #pragma omp parallel for num_threads(240) schedule(dynamic,10)
        for(int testId = 0; testId<fb_h_size/2; testId++)
        {
            int e1 = fb_h[testId*2];
            int e2 = fb_l[testId*2];
            int rel = fb_r[testId*2];
            int rel_rev = rel + relation_num/2;

            vector<pair<int,double> > rel_res;
            rel_res.resize(relation_num/2);

            for(int relationId = 0; relationId<relation_num/2; relationId++)
            {
                double sum = 0;
                sum+=calc_sum(e1,e2,relationId);
                sum+=calc_sum(e2,e1,relationId+relation_num/2);
                rel_res[relationId] = make_pair(relationId,sum);
            }

            sort(rel_res.begin(),rel_res.end(),cmp);

            int rel_raw_num = 0,rel_filter_num = 0;
            for(int i = 0; i<relation_num/2; i++)
            {
                if(rel_res[i].first != rel)
                {
                    rel_raw_num++;
                    if(ok[make_pair(e1,rel_res[i].first)].count(e2) == 0)
                    {
                        rel_filter_num++;
                    }
                }
                else
                {
                    break;
                }
            }//for(int i = 0;i<rerank_num;i++)

            cout<<rel_raw_num<<"--"<<rel_filter_num<<endl;
			
			rel_raw_sum+=rel_raw_num;
            rel_filter_sum+=rel_filter_num;
            if(rel_raw_num+1<=10)
            {
            	rel_raw_hit++;
            }
            if(rel_filter_num+1<=10)
            {
            	rel_filter_hit++;
            }
            /*if(testId%10 == 0)
                cout<<testId<<"--"<<fb_h_size/2<<endl;*/
        }//for(int testId = 0;testId<fb_h_size/2;testId++)
		cout<<"***mean rank***"<<endl;	
        cout<<"rel_raw : "<<rel_raw_sum/(fb_h_size/2)<<endl;
        cout<<"rel_filter : "<<rel_filter_sum/(fb_h_size/2)<<endl;
        
        cout<<"***hits@10***"<<endl;
        cout<<"rel_raw : "<<rel_raw_hit/(fb_h_size/2)<<endl;
        cout<<"rel_filter : "<<rel_filter_hit/(fb_h_size/2)<<endl;
       
        cout<<"relation prediction over!"<<endl;
    }

    double calc_kb_PTransR(int e1,int e2,int rel)
    {
        double sum=0;
        for (int ii=0; ii<m; ii++)
        {
            double tmp1 = 0, tmp2 = 0;
            for (int jj=0; jj<n; jj++)
            {
                tmp1 +=relation_weight[rel]*A[rel][jj][ii]*entity_vec[e1][jj];
                tmp2 +=relation_weight[rel]*A[rel][jj][ii]*entity_vec[e2][jj];
            }

            double tmp = tmp2 - tmp1 - relation_vec[rel][ii];
            if (L1_flag)
            {
                sum+=-fabs(tmp);
            }
            else
            {
                sum+=-sqr(tmp);
            }
        }
        return sum;
    }

    double calc_path(int rel,vector<int> rel_path)
    {
        double sum=0;
        for (int ii=0; ii<n; ii++)
        {
            double tmp = relation_vec[rel][ii];
            for (int j=0; j<rel_path.size(); j++)
            {
                tmp-=relation_vec[rel_path[j]][ii];
            }
            if (L1_flag)
            {
                sum+=-fabs(tmp);
            }
            else
            {
                sum+=-sqr(tmp);
            }
        }
        return 10+sum;
    }

    double calc_sum(int e1,int e2,int rel)
    {
        double sum = 0;
        sum+=calc_kb_PTransR(e1,e2,rel);
        if(fb_path.count(make_pair(e1,e2))>0)
        {
            if (fb_path[make_pair(e1,e2)].size()>0)
            {
                for (int path_id = 0; path_id<fb_path[make_pair(e1,e2)].size(); path_id++)
                {
                    vector<int> rel_path = fb_path[make_pair(e1,e2)][path_id].first;
                    double pr = fb_path[make_pair(e1,e2)][path_id].second;
                    double pr_path = 0;
                    if (path_confidence.count(make_pair(rel_path,rel))>0)
                        pr_path = path_confidence[make_pair(rel_path,rel)];
                    pr_path = 0.99 * pr_path + 0.01;
                    sum+=calc_path(rel,rel_path)*pr*pr_path;
                }
            }
        }
        return sum;
    }

};

Test test;
void prepare()
{
    FILE* f1 = fopen("data/entity2id.txt","r");
    FILE* f2 = fopen("data/relation2id.txt","r");
    int x;
    while (fscanf(f1,"%s%d",buf,&x)==2)
    {
        string st=buf;
        entity2id[st]=x;
        id2entity[x]=st;
        entity_num++;
    }
    cout<<"scanf entity2id.txt over!"<<endl;
    while (fscanf(f2,"%s%d",buf,&x)==2)
    {
        string st=buf;
        relation2id[st]=x;
        id2relation[x]=st;
        id2relation[x+REL_NUM] = "-"+st;
        relation_num++;
    }
    relation_num*=2;
    cout<<"scanf relation2id.txt over!"<<endl;

    FILE* f_kb = fopen("data/test_pra.txt","r");
    while (fscanf(f_kb,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb,"%s",buf);
        string s2=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        int e1 = entity2id[s1];
        int e2 = entity2id[s2];
        int rel;
        fscanf(f_kb,"%d",&rel);
        fscanf(f_kb,"%d",&x);
        vector<pair<vector<int>,double> > b;
        b.clear();
        for (int i = 0; i<x; i++)
        {
            int y,z;
            vector<int> rel_path;
            rel_path.clear();
            fscanf(f_kb,"%d",&y);
            for (int j=0; j<y; j++)
            {
                fscanf(f_kb,"%d",&z);
                rel_path.push_back(z);
            }
            double pr;
            fscanf(f_kb,"%lf",&pr);
            b.push_back(make_pair(rel_path,pr));
        }

        ok[make_pair(e1,rel)][e2]=1;
        test.add(e1,e2,rel,b);
        vector<pair<vector<int>,double> > ().swap(b);
    }

    FILE* f_path = fopen("data/path2.txt","r");
    while (fscanf(f_path,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_path,"%s",buf);
        string s2=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        int e1 = entity2id[s1];
        int e2 = entity2id[s2];
        fscanf(f_path,"%d",&x);
        vector<pair<vector<int>,double> > b;
        b.clear();
        for (int i = 0; i<x; i++)
        {
            int y,z;
            vector<int> rel_path;
            rel_path.clear();
            fscanf(f_path,"%d",&y);
            for (int j=0; j<y; j++)
            {
                fscanf(f_path,"%d",&z);
                rel_path.push_back(z);
            }
            double pr;
            fscanf(f_path,"%lf",&pr);
            b.push_back(make_pair(rel_path,pr));
        }
        test.add(e1,e2,-1,b);
    }
    FILE*file_train = fopen("data/train.txt","r");
    while(fscanf(file_train,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(file_train,"%s",buf);
        string s2=buf;
        fscanf(file_train,"%s",buf);
        string s3=buf;
        int e1 = entity2id[s1];
        int e2 = entity2id[s2];
        int rel = relation2id[s3];
        ok[make_pair(e1,rel)][e2] = 1;
        //ok[make_pair(e2,rel+relation_num/2)][e1] = 1;
    }
    fclose(file_train);

    FILE*file_valid = fopen("data/valid.txt","r");
    while(fscanf(file_valid,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(file_valid,"%s",buf);
        string s2=buf;
        fscanf(file_valid,"%s",buf);
        string s3=buf;
        int e1 = entity2id[s1];
        int e2 = entity2id[s2];
        int rel = relation2id[s3];
        ok[make_pair(e1,rel)][e2] = 1;
        //ok[make_pair(e2,rel+relation_num/2)][e1] = 1;
    }
    fclose(file_valid);

    cout<<"relation_num="<<relation_num<<endl;
    cout<<"entity_num="<<entity_num<<endl;

    FILE* f_confidence = fopen("data/confidence.txt","r");
    while (fscanf(f_confidence,"%d",&x)==1)
    {
        int s ;
        vector<int> rel_path;
        rel_path.clear();
        for (int i=0; i<x; i++)
        {
            fscanf(f_confidence,"%d",&s);
            rel_path.push_back(s);
        }
        fscanf(f_confidence,"%d",&x);
        for (int i=0; i<x; i++)
        {
            int y;
            double pr;
            fscanf(f_confidence,"%d%lf",&y,&pr);
            path_confidence[make_pair(rel_path,y)] = pr;
        }
    }
    cout<<"scanf confidence.txt over!"<<endl;
    FILE* f_relation_weight = fopen("data/relation_weight.txt","r");
    relation_type.resize(relation_num);
    int relation_id;
    while(fscanf(f_relation_weight,"%d",&relation_id)==1)
    {
        double t_left,t_right,t_weight;
        fscanf(f_relation_weight,"%lf%lf%lf",&t_left,&t_right,&t_weight);
        left_mean[relation_id] = t_left;
        right_mean[relation_id] = t_right;
        relation_weight[relation_id] = t_weight;

        left_mean[relation_id+relation_num/2] = t_right;
        right_mean[relation_id+relation_num/2] = t_left;
        relation_weight[relation_id+relation_num/2] = t_weight;
        if (t_left < 1.5 && t_right < 1.5)
        {
            relation_type[relation_id] = 0;
            relation_type[relation_id+relation_num/2] = 0;
        }
        else if(t_left >= 1.5 && t_right < 1.5)
        {
            relation_type[relation_id] = 1;
            relation_type[relation_id+relation_num/2] = 2;
        }
        else if(t_left < 1.5 && t_right >=1.5)
        {
            relation_type[relation_id] = 2;
            relation_type[relation_id+relation_num/2] = 1;
        }
        else
        {
            relation_type[relation_id] = 3;
            relation_type[relation_id+relation_num/2] = 3;
        }
    }
    cout<<"scanf relation_weight.txt over!"<<endl;
    fclose(f_relation_weight);
    fclose(f_confidence);
    fclose(f_kb);
}

int main(int argc,char**argv)
{
    srand((unsigned) time(NULL));
    prepare();
    test.run();
}
