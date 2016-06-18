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

char buf[10000],buf1[10000],buf2[10000];

map<string,int> relation2id,entity2id;
map<int,string> id2entity,id2relation;
map<int,int> entity2num;
int entity_num,relation_num;
map<pair<pair<int,int>,int>,int>  all;

int main(int argc,char**argv)
{

    /* FILE* f1 = fopen("../experiment_data/entity2id.txt","r");
    FILE* f2 = fopen("../experiment_data/relation2id.txt","r");
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
        relation_num++;
    }
    fclose(f1);
    fclose(f2);

    cout<<"entity_num = "<<entity_num<<" relation_num  "<<relation_num<<endl;

    FILE* file_triplet = fopen("../experiment_data/triplet.txt","w");
    FILE* file_data = fopen("../experiment_data/dat.txt","r");
    while(fscanf(file_data,"%s\t%s\t%s",buf,buf1,buf2) == 3)
    {
        string s1 = buf;
        string s2 = buf1;
        string s3 = buf2;

        int e1 = entity2id[s1];
        int e2 = entity2id[s3];
        int rel = relation2id[s2];

        if(all.count(make_pair(make_pair(e1,e2),rel)) == 0)
        {
            fprintf(file_triplet,"%s\t%s\t%s\n",s1.c_str(),s3.c_str(),s2.c_str());
        }
        all[make_pair(make_pair(e1,e2),rel)] = 1;
    }
    cout<<all.size()<<endl;
    fclose(file_triplet);
    fclose(file_data);
    cout<<"over!"<<endl;*/
    int e1,e2;
    map<pair<int,int>,int> ok;
    FILE* file_e1_e2 = fopen("../experiment_data/e1_e2.txt","r");
    FILE* file_e2_e1 = fopen("../experiment_data/e2_e1.txt","w");
    while(fscanf(file_e1_e2,"%d%d",&e1,&e2) == 2)
    {
        if(ok.count(make_pair(e1,e2)) == 0)
        {
            fprintf(file_e2_e1,"%d\t%d\n",e1,e2);
        }
        ok[make_pair(e1,e2)] = 1;
    }
    cout<<ok.size()<<endl;
    fclose(file_e1_e2);
    fclose(file_e2_e1);
    cout<<"over!"<<endl;
}


