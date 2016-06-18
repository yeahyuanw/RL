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
using namespace std;


#define pi 3.1415926535897932384626433832795


map<vector<int>,string> path2s;
map<pair<string,int>,double>  path_confidence;

bool L1_flag=false;
int entity_space = 50;
int relation_space = 50;

double rate_train = 0.001;
int eval_train = 500;
string resultfile= "result_001_1_50_L2";

double margin_train = 1;
string version="bern";
int REL_NUM = 6561;
char buf[100000];
int relation_num,entity_num;
map<string,int> relation2id,entity2id;
map<int,string> id2entity,id2relation;
vector<vector<pair<int,int> > > path;
map<int,double> left_mean,right_mean,relation_weight;

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

class Train
{

public:
    map<pair<int,int>, map<int,int> > ok;
    void add(int x,int y,int z, vector<pair<vector<int>,double> > path_list)
    {
        fb_h.push_back(x);
        fb_r.push_back(z);
        fb_l.push_back(y);
        fb_path.push_back(path_list);
        ok[make_pair(x,z)][y]=1;
    }
    void run()
    {
        n = entity_space;
        m=relation_space;
        rate = rate_train;

        cout<<"n="<<n<<" m="<<m<<" rate="<<rate<<endl;

        relation_vec.resize(relation_num);
        for (int i=0; i<relation_vec.size(); i++)
            relation_vec[i].resize(n);
        entity_vec.resize(entity_num);
        for (int i=0; i<entity_vec.size(); i++)
            entity_vec[i].resize(n);
        relation_tmp.resize(relation_num);
        for (int i=0; i<relation_tmp.size(); i++)
            relation_tmp[i].resize(n);
        entity_tmp.resize(entity_num);
        for (int i=0; i<entity_tmp.size(); i++)
            entity_tmp[i].resize(n);
        for (int i=0; i<relation_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                relation_vec[i][ii] = randn(0,1.0/n,-6/sqrt(n),6/sqrt(n));
        }
        for (int i=0; i<entity_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                entity_vec[i][ii] = randn(0,1.0/n,-6/sqrt(n),6/sqrt(n));
            norm(entity_vec[i]);
        }

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
                    if (ii==jj)
                        A[i][jj][ii]=1;
                    else
                        A[i][jj][ii]=0;
                }
            }
        }
        bfgs();
    }

private:
    int n,m;
    double rate;//learning rate
    double belta;
    vector<int> fb_h,fb_l,fb_r;
    vector<vector<pair<vector<int>,double> > >fb_path;
    vector<vector<int> > feature;
    vector<vector<double> > relation_vec,entity_vec;
    vector<vector<double> > relation_tmp,entity_tmp;
    vector<vector<vector<double> > > A,A_tmp;

    double norm(vector<double> &a)
    {
        double x = vec_len(a);
        if (x>1)
            for (int ii=0; ii<a.size(); ii++)
                a[ii]/=x;
        return 0;
    }

    void norm(vector<double> &a, vector<vector<double> > &AA,int rel)
    {
        vector<double> temp;
        temp.resize(m);
        for (int ii=0; ii<m; ii++)
        {
            double ttt = 0;
            for (int jj=0; jj<n; jj++)
                ttt+=AA[jj][ii]*a[jj]*relation_weight[rel];
            temp[ii] = ttt;
        }

        while(true)
        {
             double x = vec_len(temp);
            if(x>1)
            {
                double lambda=1;
                for (int ii=0; ii<m; ii++)
                {
                    double tmp = 0;
                    for (int jj=0; jj<n; jj++)
                        tmp+=AA[jj][ii]*a[jj];
                    tmp*=relation_weight[rel];
                    tmp*=2;

                    double ttt = 0;
                    for (int jj=0; jj<n; jj++)
                    {
						double AA_tmp = AA[jj][ii];
                        AA[jj][ii]-=rate*lambda*tmp*a[jj]*relation_weight[rel];
                        a[jj]-=rate*lambda*tmp*AA_tmp*relation_weight[rel];
                        ttt+=AA[jj][ii]*a[jj]*relation_weight[rel];
                    }
                    temp[ii] = ttt;
                }
            }
            else
            {
                break;
            }
        }//while
    }

    int rand_max(int x)
    {
        int res = (rand()*rand())%x;
        while (res<0)
            res+=x;
        return res;
    }

    void bfgs()
    {
        double margin = margin_train;

        int nbatches=100;
        int neval = eval_train;

        cout<<"margin="<<' '<<margin<<" neval= "<<neval<<" nbatches="<<nbatches<<endl;

        int batchsize = fb_h.size()/nbatches;
        A_tmp=A;
        relation_tmp=relation_vec;
        entity_tmp = entity_vec;
        map<string, int> fb_count;
        vector<double> eval_res;

        cout<<"start evaluate!"<<endl;
        for (int eval=0; eval<neval; eval++)
        {
            double res=0;
            for (int batch = 0; batch<nbatches; batch++)
            {
                #pragma omp parallel for schedule(static,10) reduction(+:res) num_threads(100)
                for (int k=0; k<batchsize; k++)
                {
                    double res_kb = 0,res_path = 0;

                    int j=rand_max(entity_num);
                    int i=rand_max(fb_h.size());
                    int e1 = fb_h[i], rel = fb_r[i], e2  = fb_l[i];

                    int rand_tmp = rand()%1000;
                    if (rand_tmp<500)
                    {
                        double l_pr = 500*right_mean[rel]/(left_mean[rel]+right_mean[rel]);
                        if ((double)rand_tmp < l_pr)
                        {
                            while (ok[make_pair(e1,rel)].count(j)>0)
                                j=rand_max(entity_num);
                            res_kb=train_kb(e1,e2,rel,e1,j,rel,margin);
                        }
                        else
                        {
                            while (ok[make_pair(j,rel)].count(e2)>0)
                                j=rand_max(entity_num);
                            res_kb=train_kb(e1,e2,rel,j,e2,rel,margin);
                        }
                    }//rand_tmp<500 if
                    else
                    {
                        int rel_neg = rand_max(relation_num);
                        while (ok[make_pair(e1,rel_neg)].count(e2)>0)
                            rel_neg = rand_max(relation_num);
                        res_kb = train_kb(e1,e2,rel,e1,e2,rel_neg,margin);
                        //norm(relation_tmp[rel_neg]);
                    }//rand_tmp<500 else
                    /**************************************************/
                    if (fb_path[i].size()>0)
                    {
                        int rel_neg = rand_max(relation_num);
                        while (ok[make_pair(e1,rel_neg)].count(e2)>0)
                            rel_neg = rand_max(relation_num);
                        for (int path_id = 0; path_id<fb_path[i].size(); path_id++)
                        {
                            vector<int> rel_path = fb_path[i][path_id].first;
                            string  s = "";
                            if (path2s.count(rel_path)==0)
                            {
                                ostringstream oss;//创建一个流
                                for (int ii=0; ii<rel_path.size(); ii++)
                                {
                                    oss<<rel_path[ii]<<" ";
                                }
                                s=oss.str();//
                                path2s[rel_path] = s;
                            }
                            s = path2s[rel_path];

                            double pr = fb_path[i][path_id].second;
                            double pr_path = 0;
                            if (path_confidence.count(make_pair(s,rel))>0)
                                pr_path = path_confidence[make_pair(s,rel)];
                            pr_path = 0.99*pr_path + 0.01;
                            res_path=train_path(rel,rel_neg,rel_path,2*margin,pr*pr_path);
                        }
                    }//if
                    res+=res_kb+res_path;
                }//for k
                A = A_tmp;
                relation_vec = relation_tmp;
                entity_vec = entity_tmp;

                if(batch%10 == 0)
                {
                    cout<<"eval:"<<eval<<" batch:"<<batch<<endl;
                }
            }
            eval_res.push_back(res);
            cout<<"eval:"<<eval<<' '<<res<<endl;
            if(eval%50 == 0 || eval == 20)
            {
                FILE* f1 = fopen((resultfile+"/A.txt"+version).c_str(),"w");
                FILE* f2 = fopen((resultfile+"/relation2vec.txt"+version).c_str(),"w");
                FILE* f3 = fopen((resultfile+"/entity2vec.txt"+version).c_str(),"w");

                for (int i=0; i<relation_num; i++)
                {
                    for (int jj=0; jj<m; jj++)
                    {
                        for (int ii=0; ii<n; ii++)
                        {
                            fprintf(f1,"%.12lf\t",A[i][jj][ii]);
                        }
                        fprintf(f1,"\n");
                    }
                }

                for (int i=0; i<relation_num; i++)
                {
                    for (int ii=0; ii<n; ii++)
                        fprintf(f2,"%.6lf\t",relation_vec[i][ii]);
                    fprintf(f2,"\n");
                }
                for (int i=0; i<entity_num; i++)
                {
                    for (int ii=0; ii<n; ii++)
                        fprintf(f3,"%.6lf\t",entity_vec[i][ii]);
                    fprintf(f3,"\n");
                }
                fclose(f1);
                fclose(f2);
                fclose(f3);
            }//if(eval%50 == 0)
        }//for eval
        FILE* f1 = fopen((resultfile+"/A.txt"+version).c_str(),"w");
        FILE* f2 = fopen((resultfile+"/relation2vec.txt"+version).c_str(),"w");
        FILE* f3 = fopen((resultfile+"/entity2vec.txt"+version).c_str(),"w");
        FILE* f4 = fopen((resultfile+"/eval_res.txt"+version).c_str(),"w");

        for (int i=0; i<relation_num; i++)
        {
            for (int jj=0; jj<m; jj++)
            {
                for (int ii=0; ii<n; ii++)
                {
                    fprintf(f1,"%.6lf\t",A[i][jj][ii]);
                }
                fprintf(f1,"\n");
            }
        }

        for (int i=0; i<relation_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                fprintf(f2,"%.6lf\t",relation_vec[i][ii]);
            fprintf(f2,"\n");
        }
        for (int i=0; i<entity_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                fprintf(f3,"%.6lf\t",entity_vec[i][ii]);
            fprintf(f3,"\n");
        }

        for (int eval = 0; eval < neval; eval++)
        {
            fprintf(f4,"%.6lf\n",eval_res[eval]);
        }

        fclose(f1);
        fclose(f2);
        fclose(f3);
        fclose(f4);
        cout<<"over!"<<endl;
    }

    double calc_kb(int e1,int e2,int rel,int same)
    {
        vector<double> e1_vec;
        e1_vec.resize(m);
        vector<double> e2_vec;
        e2_vec.resize(m);
        for (int ii=0; ii<m; ii++)
        {
            for (int jj=0; jj<n; jj++)
            {
                e1_vec[ii]+=A[rel][jj][ii]*entity_vec[e1][jj]*relation_weight[rel];
                e2_vec[ii]+=A[rel][jj][ii]*entity_vec[e2][jj]*relation_weight[rel];
            }
        }
        double sum=0;
        if (L1_flag)
            for (int ii=0; ii<m; ii++)
                sum+=fabs(e2_vec[ii]-e1_vec[ii]-same*relation_vec[rel][ii]);
        else
            for (int ii=0; ii<m; ii++)
                sum+=sqr(e2_vec[ii]-e1_vec[ii]-same*relation_vec[rel][ii]);
        return sum;
    }
    void gradient_one(int e1, int e2, int rel, int belta,int same)
    {

        vector<vector<double> > temp_AA_tmp = A_tmp[rel];
        vector<double>  temp_entity_e1 = entity_tmp[e1];
        vector<double>  temp_entity_e2 = entity_tmp[e2];
        vector<double>  temp_relation = relation_tmp[rel];

        for (int ii=0; ii<m; ii++)
        {
            double tmp1 = 0, tmp2 = 0;
            for (int jj=0; jj<n; jj++)
            {
                tmp1+=A[rel][jj][ii]*entity_vec[e1][jj]*relation_weight[rel];
                tmp2+=A[rel][jj][ii]*entity_vec[e2][jj]*relation_weight[rel];
            }
            double x = 2*(tmp2-tmp1-relation_vec[rel][ii]);
            if (L1_flag)
                if (x>0)
                    x=1;
                else
                    x=-1;
            for (int jj=0; jj<n; jj++)
            {
                temp_AA_tmp[jj][ii]-=belta*rate*x*(entity_vec[e1][jj]-entity_vec[e2][jj])*relation_weight[rel];
                temp_entity_e1[jj]-=belta*rate*x*A[rel][jj][ii]*relation_weight[rel];
                temp_entity_e2[jj]+=belta*rate*x*A[rel][jj][ii]*relation_weight[rel];
            }
            temp_relation[ii]-=same*belta*rate*x;
        }

        norm(temp_relation);
        norm(temp_entity_e1);
        norm(temp_entity_e2);
        norm(temp_entity_e1,temp_AA_tmp,rel);
        norm(temp_entity_e2,temp_AA_tmp,rel);

        A_tmp[rel] = temp_AA_tmp;
        entity_tmp[e1] = temp_entity_e1;
        entity_tmp[e2] = temp_entity_e2;
        relation_tmp[rel] = temp_relation;
    }

    void gradient_kb(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b)
    {
        gradient_one(e1_a,e2_a,rel_a,-1,1);
        gradient_one(e1_b,e2_b,rel_b,1,1);
    }

    double calc_path(int r1,vector<int> rel_path)
    {
        double sum=0;
        for (int ii=0; ii<n; ii++)
        {
            double tmp = relation_vec[r1][ii];
            for (int j=0; j<rel_path.size(); j++)
                tmp-=relation_vec[rel_path[j]][ii];
            if (L1_flag)
                sum+=fabs(tmp);
            else
                sum+=sqr(tmp);
        }
        return sum;
    }

    void gradient_path(int r1,vector<int> rel_path, double belta)
    {

        vector<double>  temp_relation_r1 = relation_tmp[r1];
        map<int,vector<double> > temp_relation_relpath;

        for (int j=0; j<rel_path.size(); j++)
                temp_relation_relpath[j] = relation_tmp[rel_path[j]];

        for (int ii=0; ii<n; ii++)
        {

            double x = relation_vec[r1][ii];
            for (int j=0; j<rel_path.size(); j++)
                x-=relation_vec[rel_path[j]][ii];
            if (L1_flag)
                if (x>0)
                    x=1;
                else
                    x=-1;
            temp_relation_r1[ii]+=belta*rate*x;
            for (int j=0; j<rel_path.size(); j++)
                temp_relation_relpath[j][ii]-=belta*rate*x;
        }
        for (int j=0; j<rel_path.size(); j++)
        {
            norm(temp_relation_relpath[j]);
            relation_tmp[rel_path[j]] = temp_relation_relpath[j];
        }

        norm(temp_relation_r1);
        relation_tmp[r1] = temp_relation_r1;
    }//void gradient_path(int r1,vector<int> rel_path, double belta)

    double train_kb(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b,double margin)
    {
        double sum1 = calc_kb(e1_a,e2_a,rel_a,1);
        double sum2 = calc_kb(e1_b,e2_b,rel_b,1);
        double sum = 0;
        if (sum1+margin>sum2)
        {
            sum=margin+sum1-sum2;
            gradient_kb(e1_a, e2_a, rel_a,e1_b, e2_b, rel_b);
        }
        return sum;
    }
    double train_path(int rel, int rel_neg, vector<int> rel_path, double margin,double x)
    {
        double sum1 = calc_path(rel,rel_path);
        double sum2 = calc_path(rel_neg,rel_path);
        double lambda = 1;
        double sum = 0;
        if (sum1+margin>sum2)
        {
            sum=x*lambda*(margin+sum1-sum2);
            gradient_path(rel,rel_path, -x*lambda);
            gradient_path(rel_neg,rel_path, x*lambda);
        }
        return sum;
    }
};

Train train;
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
    while (fscanf(f2,"%s%d",buf,&x)==2)
    {
        string st=buf;
        relation2id[st]=x;
        id2relation[x]=st;
        id2relation[x+REL_NUM] = "-"+st;
        relation_num++;
    }
    FILE* f_kb = fopen("data/train_pra.txt","r");
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
        //cout<<e1<<' '<<e2<<' '<<rel<<' '<<b.size()<<endl;
        train.add(e1,e2,rel,b);
    }
    relation_num*=2;

    cout<<"relation_num="<<relation_num<<endl;
    cout<<"entity_num="<<entity_num<<endl;

    FILE* f_confidence = fopen("data/confidence.txt","r");
    while (fscanf(f_confidence,"%d",&x)==1)
    {
        string s = "";
        for (int i=0; i<x; i++)
        {
            fscanf(f_confidence,"%s",buf);
            s = s + string(buf)+" ";
        }
        fscanf(f_confidence,"%d",&x);
        for (int i=0; i<x; i++)
        {
            int y;
            double pr;
            fscanf(f_confidence,"%d%lf",&y,&pr);
            path_confidence[make_pair(s,y)] = pr;
        }
    }
    fclose(f_confidence);
    fclose(f_kb);

    FILE* f_relation_weight = fopen("data/relation_weight.txt","r");
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
    }
    cout<<"prepare over!"<<endl;
    fclose(f_relation_weight);
}

int main(int argc,char**argv)
{
    srand((unsigned) time(NULL));
    prepare();
    train.run();
}
