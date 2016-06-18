
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

const char* file_train = "../experiment_data/train.txt";
const char* file_change = "../experiment_data/test.txt";

char buf[100000],buf1[100000];
int relation_num,entity_num,test_triplet_num,train_triplet_num;
map<string,int> relation2id,entity2id;
map<int,string> id2entity,id2relation;

map<int,int> entity2num;


map<int,map<int,vector<int> > > left_entity,right_entity;

//relation_tphr是某一关系r，头实体对应的尾实体平均数
//relation_hptr是某一关系r，尾实体对应的头实体平均数
map<int,float> relation_tphr,relation_hptr;

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
    FILE* f_kb = fopen(file_train,"r");
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
        entity2num[entity2id[s1]]++;
        entity2num[entity2id[s2]]++;
		//每个关系左实体对应的右实体个数
        left_entity[relation2id[s3]][entity2id[s1]].push_back(entity2id[s2]);
		//每个关系右实体对应的左实体个数
        right_entity[relation2id[s3]][entity2id[s2]].push_back(entity2id[s1]);
		train_triplet_num++;
    }
	cout<<"train_triplet_num:"<<train_triplet_num<<endl;

	FILE* relation_11 = fopen("../experiment_data/relation_1_1.txt","w");
	FILE* relation_1N = fopen("../experiment_data/relation_1_n.txt","w");
	FILE* relation_N1 = fopen("../experiment_data/relation_n_1.txt","w");
	FILE* relation_NN = fopen("../experiment_data/relation_n_n.txt","w");

	FILE* relation_weight = fopen("../experiment_data/relation_weight.txt","w");

	map<int,map<int,vector<int> > >::iterator left_iter,right_iter;
	for (left_iter = left_entity.begin(),right_iter = right_entity.begin();
		left_iter != left_entity.end() && right_iter != right_entity.end();
	    left_iter++,right_iter++)
	{
		map<int,vector<int> > left_size = left_iter->second;
		map<int,vector<int> > right_size = right_iter->second;

		int left_sum = 0,right_sum = 0;

		for (map<int,vector<int> >::iterator iter = left_size.begin();
			iter != left_size.end();
			iter++)
		{
			left_sum += iter->second.size();
		}

		for (map<int,vector<int> >::iterator iter = right_size.begin();
			iter != right_size.end();
			iter++)
		{
			right_sum += iter->second.size();
		}
		if (left_iter->first == right_iter->first)
		{
			relation_tphr[left_iter->first] = ((float)left_sum)/left_size.size();
			relation_hptr[right_iter->first] = ((float)right_sum)/right_size.size();
			float t1 = relation_tphr[left_iter->first];
			float t2 = relation_hptr[right_iter->first];
			double ww = 1/(log(t1+t2)/log(10))/3.321928;

			fprintf(relation_weight,"%d\t%f\t%f\t%lf\n",left_iter->first,t1,t2,ww);

			if (t1 < 1.5 && t2 < 1.5)
			{
				fprintf(relation_11,"%d\t%f\t%f\t%lf\n",left_iter->first,t1,t2,ww);
			}
			else if(t1 >= 1.5 && t2 < 1.5){
				fprintf(relation_1N,"%d\t%f\t%f\t%lf\n",left_iter->first,t1,t2,ww);
			}
			else if(t1 < 1.5 && t2 >=1.5){
				fprintf(relation_N1,"%d\t%f\t%f\t%lf\n",left_iter->first,t1,t2,ww);
			}
			else{
				fprintf(relation_NN,"%d\t%f\t%f\t%lf\n",left_iter->first,t1,t2,ww);
			}
		}
		else
		{
			cout<<"relation not match"<<endl;
			exit(0);
		}
	}

	fclose(f_kb);
	fclose(relation_11);
	fclose(relation_1N);
	fclose(relation_N1);
	fclose(relation_NN);
	fclose(relation_weight);

	FILE* file_11 = fopen("../experiment_data/triplet_1_1.txt","w");
	FILE* file_1N = fopen("../experiment_data/triplet_1_n.txt","w");
	FILE* file_N1 = fopen("../experiment_data/triplet_n_1.txt","w");
	FILE* file_NN = fopen("../experiment_data/triplet_n_n.txt","w");

	FILE* f_kb_text = fopen(file_change,"r");
	while(fscanf(f_kb_text,"%s",buf)==1)
	{
		test_triplet_num++;
		string s1=buf;
        fscanf(f_kb_text,"%s",buf);
        string s2=buf;
        fscanf(f_kb_text,"%s",buf);
        string s3=buf;

		const char *ss1 = s1.c_str();
		const char *ss2 = s2.c_str();
		const char *ss3 = s3.c_str();

		const char * format = "%s\t%s\t%s\n";

		float tphr_mean,hptr_mean;

		if(relation_tphr.count(relation2id[s3]) == 0)
		{
			cout<<s3+" is covered!"<<endl;
			FILE* file_miss = fopen("../experiment_data/miss.txt","w");
			fprintf(file_miss,"%s\n",s3.c_str());
			fclose(file_miss);
			continue;
		}
		else
		{
			tphr_mean = relation_tphr[relation2id[s3]];
			hptr_mean = relation_hptr[relation2id[s3]];
		}

		//cout<<s1<<"\t"<<s2<<"\t"<<s3<<"\t"<<tphr_mean<<"\t"<<hptr_mean<<endl;
		//1-1
		if(tphr_mean < 1.5 && hptr_mean < 1.5)
		{
			if (file_11 != NULL)
			{
				fprintf(file_11,format,ss1,ss2,ss3);
			}
			else
			{
				cout<<"error:file_11 is NULL."<<endl;
				break;
			}
		}
		//N-1
		else if(tphr_mean < 1.5 && hptr_mean >= 1.5)
		{
			if (file_N1 != NULL)
			{
				fprintf(file_N1,format,ss1,ss2,ss3);
			}
			else
			{
				cout<<"error:file_N1 is NULL."<<endl;
				break;
			}
		}
		//1-N
		else if(tphr_mean >= 1.5 && hptr_mean < 1.5)
		{
			if (file_1N != NULL)
			{
				fprintf(file_1N,format,ss1,ss2,ss3);
			}
			else
			{
				cout<<"error:file_1N is NULL."<<endl;
				break;
			}
		}
		//N-N
		else
		{
			if (file_NN != NULL)
			{
				fprintf(file_NN,format,ss1,ss2,ss3);
			}
			else
			{
				cout<<"error:file_NN is NULL."<<endl;
				break;
			}
		}
	}

	cout<<"test_triplet_num:"<<test_triplet_num<<endl;

	fclose(file_11);
	fclose(file_1N);
	fclose(file_N1);
	fclose(file_NN);
	fclose(f_kb_text);
}


