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

string file_path = "result_001_1_20_L1";
int entity_hit_n = 10;
int rel_hit_n = 1;
int rel_num = 1345;
int main(int argc,char**argv)
{
    char buf[100000];
    int relation_num,entity_num;
    map<string,int> relation2id,entity2id;
    map<int,string> id2entity,id2relation;
    vector<vector<pair<vector<int>,double> > >fb_path;
    vector<int> fb_rel;

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

    FILE* f_kb = fopen("../experiment_data/train_pra.txt","r");
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
        fb_rel.push_back(rel);
        fb_path.push_back(b);
    }
    fclose(f_kb);
    
    map<int,double> left_mean,right_mean;
    FILE* f_relation_weight = fopen("experiment_data/relation_weight.txt","r");
    int relation_id;
    while(fscanf(f_relation_weight,"%d",&relation_id)==1)
    {
        double t_left,t_right,t_weight;
        fscanf(f_relation_weight,"%lf%lf%lf",&t_left,&t_right,&t_weight);
        left_mean[relation_id] = t_left;
        right_mean[relation_id] = t_right;

    }
    cout<<"prepare over!"<<endl;
    fclose(f_relation_weight);
    
    int test_num = fb_path.size()/2;
    int path_triplet_num = 0;
    
    int num0 = 0;
    int num = 0;
    for(int id = 0;id<test_num;id++)
    {
        int ttt = id*2;
        int rel = fb_rel[ttt];
        if(fb_path[ttt].size() == 0)
        {
        	if(left_mean[rel] < 1.5 && right_mean[rel] < 1.5)
        	{
        		num0++;
        	}
        }
        if(left_mean[rel]<1.5 && right_mean[rel]<1.5)
        {
        		num++;
        }
    }
    cout<<num0<<endl;
    cout<<num<<endl;
}


