#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<cmath>
#include<cstdlib>
using namespace std;

string file_path = "result_1_1_100_L1";
int rel_num = 1345;
int main(int argc,char**argv)
{
    char buf[100000];
    int relation_num,entity_num;
    map<string,int> relation2id,entity2id;
    map<int,string> id2entity,id2relation;
    vector<vector<pair<vector<int>,double> > >fb_path;

    FILE* f1 = fopen("../experiment_data/entity2id.txt","r");
    FILE* f2 = fopen("../experiment_data/relation2id.txt","r");
    int x;
    while (fscanf(f1,"%s%d",buf,&x)==2)
    {
        string st=buf;
        entity2id[st]=x;
        id2entity[x]=st;
        entity_num++;
    }
    fclose(f1);

    while (fscanf(f2,"%s%d",buf,&x)==2)
    {
        string st=buf;
        relation2id[st]=x;
        id2relation[x]=st;
        id2relation[x+rel_num] = "-"+st;
        relation_num++;
    }
    fclose(f2);
    relation_num*=2;

    FILE* f_kb = fopen("../experiment_data/test_pra.txt","r");
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

        fb_path.push_back(b);
    }
    fclose(f_kb);
    FILE* f_l_raw = fopen(("../"+file_path+"/l_raw_rank.txt").c_str(),"r");
    FILE* f_l_filter = fopen(("../"+file_path+"/l_filter_rank.txt").c_str(),"r");
    FILE* f_r_raw = fopen(("../"+file_path+"/r_raw_rank.txt").c_str(),"r");
    FILE* f_r_filter = fopen(("../"+file_path+"/r_filter_rank.txt").c_str(),"r");

    int testId = 0;
    vector<int> l_raw,l_filter,r_raw,r_filter,type;
    while (fscanf(f_l_raw,"%d",&testId)==1)
    {
    int t = 0,res = 0;
    fscanf(f_l_raw,"%d",&t);
    fscanf(f_l_raw,"%d",&res);
    type.push_back(t);
    l_raw.push_back(res);
    }

    while (fscanf(f_l_filter,"%d",&testId)==1)
    {
    int t = 0,res = 0;
    fscanf(f_l_filter,"%d",&t);
    fscanf(f_l_filter,"%d",&res);
    l_filter.push_back(res);
    }

    while (fscanf(f_r_raw,"%d",&testId)==1)
    {
    int t = 0,res = 0;
    fscanf(f_r_raw,"%d",&t);
    fscanf(f_r_raw,"%d",&res);
    r_raw.push_back(res);
    }

    while (fscanf(f_r_filter,"%d",&testId)==1)
    {
    int t = 0,res = 0;
    fscanf(f_r_filter,"%d",&t);
    fscanf(f_r_filter,"%d",&res);
    r_filter.push_back(res);
    }

    fclose(f_l_raw);
    fclose(f_l_filter);
    fclose(f_r_raw);
    fclose(f_r_filter);

    int test_num = fb_path.size()/2;
    int path_triplet_num = 0;
    int l_raw_sum = 0,l_filter_sum = 0,r_raw_sum = 0,r_filter_sum = 0;
    for(int id = 0;id<test_num;id++)
    {
        int ttt = id*2;
        if(fb_path[ttt].size()>0)
        {
            path_triplet_num++;
            l_raw_sum +=l_raw[id];
            l_filter_sum +=l_filter[id];
            r_raw_sum +=r_raw[id];
            r_filter_sum +=r_filter[id];

        }
    }

    cout<<"path_triplet_num = "<<path_triplet_num<<"  rest: "<<test_num-path_triplet_num<<endl;
    cout<<"l_raw_sum = "<<l_raw_sum/path_triplet_num<<"  l_filter_sum = "<<l_filter_sum/path_triplet_num<<endl;
    cout<<"r_raw_sum = "<<r_raw_sum/path_triplet_num<<"  r_filter_sum = "<<r_filter_sum/path_triplet_num<<endl;
    cout<<"raw : "<<(l_raw_sum+r_raw_sum)/(2*path_triplet_num)<<"  filter : "<<(l_filter_sum+r_filter_sum)/(2*path_triplet_num)<<endl;
}


