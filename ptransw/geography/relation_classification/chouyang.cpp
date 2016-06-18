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

int train_11 = 55940,test_11 = 6839,valid_11 = 5792;
int train_1n = 44,test_1n = 6, valid_1n = 4;
int train_n1 = 24812,test_n1 = 3033,valid_n1 = 2569;
int train_nn = 19,test_nn = 3,valid_nn = 2;

map<string,int> relation2id,entity2id;
map<int,string> id2entity,id2relation;
map<int,int> entity2num;
int entity_num,relation_num;
vector<int> fb_h_11,fb_t_11,fb_r_11;
vector<int> fb_h_1n,fb_t_1n,fb_r_1n;
vector<int> fb_h_n1,fb_t_n1,fb_r_n1;
vector<int> fb_h_nn,fb_t_nn,fb_r_nn;
map<pair<pair<int,int>,int>,int>  train,test,all;
int rand_max(int x)
{
    int res = (rand()*rand())%x;
    while (res<0)
        res+=x;
    return res;
}

int main(int argc,char**argv)
{
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

    FILE* file_1_1 = fopen("../experiment_data/triplet_1_1.txt","r");
    FILE* file_1_n = fopen("../experiment_data/triplet_1_n.txt","r");
    FILE* file_n_1 = fopen("../experiment_data/triplet_n_1.txt","r");
    FILE* file_n_n = fopen("../experiment_data/triplet_n_n.txt","r");

    int triplet_num_11 = 0;
    while(fscanf(file_1_1,"%s\t%s\t%s",buf,buf1,buf2) == 3)
    {
        string s1 = buf;
        string s2 = buf1;
        string s3 = buf2;

        int e1 = entity2id[s1];
        int e2 = entity2id[s2];
        int rel = relation2id[s3];

        triplet_num_11++;

        fb_h_11.push_back(e1);
        fb_t_11.push_back(e2);
        fb_r_11.push_back(rel);
    }
    cout<<"triplet_num_11 = "<<triplet_num_11<<endl;

    int triplet_num_1n = 0;
    while(fscanf(file_1_n,"%s\t%s\t%s",buf,buf1,buf2) == 3)
    {
        string s1 = buf;
        string s2 = buf1;
        string s3 = buf2;

        int e1 = entity2id[s1];
        int e2 = entity2id[s2];
        int rel = relation2id[s3];

        triplet_num_1n++;

        fb_h_1n.push_back(e1);
        fb_t_1n.push_back(e2);
        fb_r_1n.push_back(rel);
    }
    cout<<"triplet_num_1n = "<<triplet_num_1n<<endl;

    int triplet_num_n1 = 0;
    while(fscanf(file_n_1,"%s\t%s\t%s",buf,buf1,buf2) == 3)
    {
        string s1 = buf;
        string s2 = buf1;
        string s3 = buf2;

        int e1 = entity2id[s1];
        int e2 = entity2id[s2];
        int rel = relation2id[s3];

        triplet_num_n1++;

        fb_h_n1.push_back(e1);
        fb_t_n1.push_back(e2);
        fb_r_n1.push_back(rel);
    }
    cout<<"triplet_num_n1 = "<<triplet_num_n1<<endl;

    int triplet_num_nn = 0;
    while(fscanf(file_n_n,"%s\t%s\t%s",buf,buf1,buf2) == 3)
    {
        string s1 = buf;
        string s2 = buf1;
        string s3 = buf2;

        int e1 = entity2id[s1];
        int e2 = entity2id[s2];
        int rel = relation2id[s3];

        triplet_num_nn++;

        fb_h_nn.push_back(e1);
        fb_t_nn.push_back(e2);
        fb_r_nn.push_back(rel);
    }
    cout<<"triplet_num_nn = "<<triplet_num_nn<<endl;

    fclose(file_1_1);
    fclose(file_1_n);
    fclose(file_n_1);
    fclose(file_n_n);

    srand((unsigned) time(NULL));

    FILE* file_train = fopen("../experiment_data/train.txt","w");
    for(int i = 0;i<train_11;i++)
    {
        int t = rand_max(fb_h_11.size());
        while(train.count(make_pair(make_pair(fb_h_11[t],fb_t_11[t]),fb_r_11[t])) > 0)
        {
            t = rand_max(fb_h_11.size());
        }
        string s1 = id2entity[fb_h_11[t]];
        string s2 = id2entity[fb_t_11[t]];
        string s3 = id2relation[fb_r_11[t]];

        fprintf(file_train,"%s\t%s\t%s\n",s1.c_str(),s2.c_str(),s3.c_str());
        train[make_pair(make_pair(fb_h_11[t],fb_t_11[t]),fb_r_11[t])] = 1;
    }

    for(int i=0;i<train_1n;i++)
    {
        int t = rand_max(fb_h_1n.size());
        while(train.count(make_pair(make_pair(fb_h_1n[t],fb_t_1n[t]),fb_r_1n[t])) > 0)
        {
            t = rand_max(fb_h_1n.size());
        }
        string s1 = id2entity[fb_h_1n[t]];
        string s2 = id2entity[fb_t_1n[t]];
        string s3 = id2relation[fb_r_1n[t]];

        fprintf(file_train,"%s\t%s\t%s\n",s1.c_str(),s2.c_str(),s3.c_str());
         train[make_pair(make_pair(fb_h_1n[t],fb_t_1n[t]),fb_r_1n[t])] = 1;
    }

    for(int i=0;i<train_n1;i++)
    {
        int t = rand_max(fb_h_n1.size());
        while(train.count(make_pair(make_pair(fb_h_n1[t],fb_t_n1[t]),fb_r_n1[t])) > 0)
        {
            t = rand_max(fb_h_n1.size());
        }

        string s1 = id2entity[fb_h_n1[t]];
        string s2 = id2entity[fb_t_n1[t]];
        string s3 = id2relation[fb_r_n1[t]];

        fprintf(file_train,"%s\t%s\t%s\n",s1.c_str(),s2.c_str(),s3.c_str());
         train[make_pair(make_pair(fb_h_n1[t],fb_t_n1[t]),fb_r_n1[t])] = 1;
    }

    for(int i=0;i<train_nn;i++)
    {
        int t = rand_max(fb_h_nn.size());
        while(train.count(make_pair(make_pair(fb_h_nn[t],fb_t_nn[t]),fb_r_nn[t])) > 0)
        {
            t = rand_max(fb_h_nn.size());
        }

        string s1 = id2entity[fb_h_nn[t]];
        string s2 = id2entity[fb_t_nn[t]];
        string s3 = id2relation[fb_r_nn[t]];

        fprintf(file_train,"%s\t%s\t%s\n",s1.c_str(),s2.c_str(),s3.c_str());
         train[make_pair(make_pair(fb_h_nn[t],fb_t_nn[t]),fb_r_nn[t])] = 1;
    }

    fclose(file_train);
    cout<<"train over!"<<endl;
    FILE* file_test = fopen("../experiment_data/test.txt","w");
    for(int i = 0;i<test_11;i++)
    {
        int t = rand_max(fb_h_11.size());
        while(train.count(make_pair(make_pair(fb_h_11[t],fb_t_11[t]),fb_r_11[t])) > 0 || test.count(make_pair(make_pair(fb_h_11[t],fb_t_11[t]),fb_r_11[t])) > 0)
        {
            t = rand_max(fb_h_11.size());
        }

        string s1 = id2entity[fb_h_11[t]];
        string s2 = id2entity[fb_t_11[t]];
        string s3 = id2relation[fb_r_11[t]];

        fprintf(file_test,"%s\t%s\t%s\n",s1.c_str(),s2.c_str(),s3.c_str());
        test[make_pair(make_pair(fb_h_11[t],fb_t_11[t]),fb_r_11[t])] = 1;
    }

    for(int i=0;i<test_1n;i++)
    {
        int t = rand_max(fb_h_1n.size());
        while(train.count(make_pair(make_pair(fb_h_1n[t],fb_t_1n[t]),fb_r_1n[t])) > 0 || test.count(make_pair(make_pair(fb_h_1n[t],fb_t_1n[t]),fb_r_1n[t])) > 0)
        {
            t = rand_max(fb_h_1n.size());
        }

        string s1 = id2entity[fb_h_1n[t]];
        string s2 = id2entity[fb_t_1n[t]];
        string s3 = id2relation[fb_r_1n[t]];

        fprintf(file_test,"%s\t%s\t%s\n",s1.c_str(),s2.c_str(),s3.c_str());
         test[make_pair(make_pair(fb_h_1n[t],fb_t_1n[t]),fb_r_1n[t])] = 1;
    }

    for(int i=0;i<test_n1;i++)
    {
        int t = rand_max(fb_h_n1.size());
        while(train.count(make_pair(make_pair(fb_h_n1[t],fb_t_n1[t]),fb_r_n1[t])) > 0 || test.count(make_pair(make_pair(fb_h_n1[t],fb_t_n1[t]),fb_r_n1[t])) > 0)
        {
            t = rand_max(fb_h_n1.size());
        }

        string s1 = id2entity[fb_h_n1[t]];
        string s2 = id2entity[fb_t_n1[t]];
        string s3 = id2relation[fb_r_n1[t]];

        fprintf(file_test,"%s\t%s\t%s\n",s1.c_str(),s2.c_str(),s3.c_str());
         test[make_pair(make_pair(fb_h_n1[t],fb_t_n1[t]),fb_r_n1[t])] = 1;
    }

    for(int i=0;i<test_nn;i++)
    {
        int t = rand_max(fb_h_nn.size());
        while(train.count(make_pair(make_pair(fb_h_nn[t],fb_t_nn[t]),fb_r_nn[t])) > 0 || test.count(make_pair(make_pair(fb_h_nn[t],fb_t_nn[t]),fb_r_nn[t])) > 0)
        {
            t = rand_max(fb_h_nn.size());
        }
        string s1 = id2entity[fb_h_nn[t]];
        string s2 = id2entity[fb_t_nn[t]];
        string s3 = id2relation[fb_r_nn[t]];

        fprintf(file_test,"%s\t%s\t%s\n",s1.c_str(),s2.c_str(),s3.c_str());
         test[make_pair(make_pair(fb_h_nn[t],fb_t_nn[t]),fb_r_nn[t])] = 1;
    }

    fclose(file_test);
    cout<<"test over!"<<endl;

    cout<<"train triplet : "<<train.size()<<"--"<<"test triplet : "<<test.size()<<endl;

    FILE* file_triplet = fopen("../experiment_data/triplet.txt","r");
    FILE* file_valid = fopen("../experiment_data/valid.txt","w");
    while(fscanf(file_triplet,"%s\t%s\t%s",buf,buf1,buf2) == 3)
    {
        string s1 = buf;
        string s2 = buf1;
        string s3 = buf2;

        int e1 = entity2id[s1];
        int e2 = entity2id[s2];
        int rel = relation2id[s3];

        if(train.count(make_pair(make_pair(e1,e2),rel)) == 0 && test.count(make_pair(make_pair(e1,e2),rel)) == 0)
        {
            fprintf(file_valid,"%s\t%s\t%s\n",s1.c_str(),s2.c_str(),s3.c_str());
        }
        all[make_pair(make_pair(e1,e2),rel)] = 1;
    }
    cout<<"all triplet : "<<all.size()<<endl;
    fclose(file_triplet);
    fclose(file_valid);
    cout<<"valid over!"<<endl;
}


