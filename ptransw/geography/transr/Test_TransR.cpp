#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<algorithm>
#include<cmath>
#include<cstdlib>
using namespace std;

bool debug=false;

string version;
string trainortest = "test";
string result_path = "result_001_1_100_L1";

map<string,int> relation2id,entity2id;
map<int,string> id2entity,id2relation;
map<string,string> mid2name,mid2type;
map<int,map<int,int> > entity2num;
map<int,int > e2num;
map<pair<string,string>,map<string,double> > rel_left,rel_right;

int relation_num,entity_num;
int m= 100;
int n= 100;

double sigmod(double x)
{
    return 1.0/(1+exp(-x));
}

double vec_len(vector<double> a)
{
    double res=0;
    for (int i=0; i<a.size(); i++)
        res+=a[i]*a[i];
    return sqrt(res);
}

void vec_output(vector<double> a)
{
    for (int i=0; i<a.size(); i++)
    {
        cout<<a[i]<<"\t";
        if (i%5==4)
            cout<<endl;
    }
    cout<<"-------------------------"<<endl;
}

double sqr(double x)
{
    return x*x;
}

char buf[100000],buf1[100000];

int my_cmp(pair<double,int> a,pair<double,int> b)
{
    return a.first>b.first;
}

double cmp(pair<int,double> a, pair<int,double> b)
{
    return a.second>b.second;
}

class Test
{
    vector<vector<double> > relation_vec,entity_vec;


    vector<int> h,l,r;
    vector<int> fb_h,fb_l,fb_r;
    vector<vector<int> > feature;
    vector<vector<vector<double> > >A;
    map<pair<int,int>, map<int,int> > ok;
    double res ;
public:
    void add(int x,int y,int z, bool flag)
    {
        if (flag)
        {
            fb_h.push_back(x);
            fb_r.push_back(z);
            fb_l.push_back(y);
        }
        ok[make_pair(x,z)][y]=1;
    }

    int rand_max(int x)
    {
        int res = (rand()*rand())%x;
        if (res<0)
            res+=x;
        return res;
    }
    double len;
    double calc_sum(int e1,int e2,int rel)
    {
        vector<double> e1_vec;
        e1_vec.resize(m);
        vector<double> e2_vec;
        e2_vec.resize(m);
        for (int ii=0; ii<m; ii++)
            for (int jj=0; jj<n; jj++)
            {
                e1_vec[ii]+=A[rel][jj][ii]*entity_vec[e1][jj];
                e2_vec[ii]+=A[rel][jj][ii]*entity_vec[e2][jj];
            }
        double sum=0;
        for (int ii=0; ii<m; ii++)
            sum+=-fabs(e2_vec[ii]-e1_vec[ii]-relation_vec[rel][ii]);
        return sum;
    }
    void run()
    {
        FILE* f1 = fopen((result_path+"/relation2vec."+version).c_str(),"r");
        FILE* f3 = fopen((result_path+"/entity2vec."+version).c_str(),"r");
        FILE* f5 = fopen((result_path+"/A."+version).c_str(),"r");
        int relation_num_fb=relation_num;
        relation_vec.resize(relation_num_fb);
        for (int i=0; i<relation_num_fb; i++)
        {
            relation_vec[i].resize(m);
            for (int ii=0; ii<m; ii++)
                fscanf(f1,"%lf",&relation_vec[i][ii]);
        }
        entity_vec.resize(entity_num);
        for (int i=0; i<entity_num; i++)
        {
            entity_vec[i].resize(n);
            for (int ii=0; ii<n; ii++)
                fscanf(f3,"%lf",&entity_vec[i][ii]);
        }
        A.resize(relation_num);
        for (int i=0; i<relation_num; i++)
        {
            A[i].resize(n);
            for (int jj=0; jj<n; jj++)
            {
                A[i][jj].resize(m);
                for (int ii=0; ii<m; ii++)
                    fscanf(f5,"%lf",&A[i][jj][ii]);
            }
        }
        fclose(f1);
        fclose(f3);
        fclose(f5);
        cout<<"entity_num = "<<entity_num<<"relation_num = "<<relation_num<<endl;
        entity_predictive();
        relation_predictive();
    }//run

    void entity_predictive()
    {
        cout<<"start entity_prediction!"<<endl;
        int test_num = fb_h.size();

        FILE* file_rank = fopen((result_path+"/entity_rank.txt").c_str(),"w");

        for(int testid = 0; testid < test_num; testid++)
        {
            int e1 = fb_h[testid];
            int e2 = fb_l[testid];
            int rel = fb_r[testid];

            vector<pair<int,double> > l_res,r_res;
            l_res.resize(entity_num);
            r_res.resize(entity_num);

            #pragma omp parallel for num_threads(4) schedule(dynamic,10)
            for(int entityid = 0; entityid<entity_num; entityid++)
            {
                double sum1 = calc_sum(entityid,e2,rel);
                double sum2 = calc_sum(e1,entityid,rel);

                l_res[entityid] = make_pair(entityid,sum1);
                r_res[entityid] = make_pair(entityid,sum2);
            }

            sort(l_res.begin(),l_res.end(),cmp);
            sort(r_res.begin(),r_res.end(),cmp);

            int l_raw_num = 0, r_raw_num = 0;
            int l_filter_num = 0,r_filter_num = 0;

            for(int i = 0; i<entity_num; i++)
            {
                if(e1 != l_res[i].first)
                {
                    l_raw_num++;
                    if(ok[make_pair(l_res[i].first,rel)].count(e2) == 0)
                    {
                        l_filter_num++;
                    }
                }
                else
                {
                    break;
                }
            }

            for(int i = 0; i<entity_num; i++)
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
                {
                    break;
                }
            }
            vector<pair<int,double> >().swap(r_res);
            vector<pair<int,double> >().swap(l_res);

            fprintf(file_rank,"%d\t%d\t%d\t%d\n",l_raw_num,r_raw_num,l_filter_num,r_filter_num);

            cout<<testid<<"--"<<test_num<<endl;
        }//for testid

        fclose(file_rank);
        cout<<"entity over!"<<endl;
    }

    void relation_predictive()
    {
        cout<<"start relation prediction!"<<endl;
        int test_num = fb_h.size();

        FILE* file_rank = fopen((result_path+"/relation_rank.txt").c_str(),"w");

        for(int testid = 0; testid < test_num; testid++)
        {
            int e1 = fb_h[testid];
            int e2 = fb_l[testid];
            int rel = fb_r[testid];

            vector<pair<int,double> > rel_res;
            rel_res.resize(relation_num);

            #pragma omp parallel for num_threads(4) schedule(dynamic,10)
            for(int relationid = 0; relationid<relation_num; relationid++)
            {
                double sum = calc_sum(e1,e2,relationid);

                rel_res[relationid] = make_pair(relationid,sum);
            }

            sort(rel_res.begin(),rel_res.end(),cmp);

            int rel_raw_num = 0,rel_filter_num = 0;

            for(int i = 0; i<relation_num; i++)
            {
                if(rel != rel_res[i].first)
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
            }

            fprintf(file_rank,"%d\t%d\n",rel_raw_num,rel_filter_num);
            cout<<testid<<"--"<<test_num<<endl;
        }//for testid

        fclose(file_rank);
        cout<<"entity over!"<<endl;

    }

};
Test test;

void prepare()
{
    FILE* f1 = fopen("../experiment_data/entity2id.txt","r");
    FILE* f2 = fopen("../experiment_data/relation2id.txt","r");
    int x;
    while (fscanf(f1,"%s%d",buf,&x)==2)
    {
        string st=buf;
        entity2id[st]=x;
        id2entity[x]=st;
        mid2type[st]="None";
        entity_num++;
    }
    while (fscanf(f2,"%s%d",buf,&x)==2)
    {
        string st=buf;
        relation2id[st]=x;
        id2relation[x]=st;
        relation_num++;
    }
    FILE* f_kb = fopen("../experiment_data/test.txt","r");
    while (fscanf(f_kb,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb,"%s",buf);
        string s2=buf;
        fscanf(f_kb,"%s",buf);
        string s3=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
            relation2id[s3] = relation_num;
            relation_num++;
        }
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],true);
    }
    fclose(f_kb);
    FILE* f_kb1 = fopen("../experiment_data/train.txt","r");
    while (fscanf(f_kb1,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb1,"%s",buf);
        string s2=buf;
        fscanf(f_kb1,"%s",buf);
        string s3=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
            relation2id[s3] = relation_num;
            relation_num++;
        }
        entity2num[relation2id[s3]][entity2id[s1]]+=1;
        entity2num[relation2id[s3]][entity2id[s2]]+=1;
        e2num[entity2id[s1]]+=1;
        e2num[entity2id[s2]]+=1;
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],false);
    }
    fclose(f_kb1);
    FILE* f_kb2 = fopen("../experiment_data/valid.txt","r");
    while (fscanf(f_kb2,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb2,"%s",buf);
        string s2=buf;
        fscanf(f_kb2,"%s",buf);
        string s3=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
            relation2id[s3] = relation_num;
            relation_num++;
        }
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],false);
    }
    fclose(f_kb2);
}


int main(int argc,char**argv)
{
    version = "bern";
    prepare();
    test.run();
}

