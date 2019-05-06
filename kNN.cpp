/*
 author: "Ami Patel"
 title: "kNN Classification"
 date: "03/03/2019"
*/

//---------------------------------------------------------------------------------------------
/* To run this .cpp file on Rstudio, make sure this file is in your current working directory 
   and then, type the command below in console:
        Rcpp::sourceCpp(file = "kNN.cpp")
 
   ** NOTE: Please make sure the R script, "ahp170730_project1.R" runs before running this .cpp file.
   This is mainly because R script will create the .csv files for C++ program. 
*/

//---------------------------------------------------------------------------------------------

#include<Rcpp.h>
#include<vector>
#include<cmath>
#include<algorithm>
#include<cstdio>
#include<cstdlib>
#include<fstream>
#include<iostream>
#include<cstring>
#include<chrono>
#include<numeric>
//-----------------------------------

using namespace std;
using namespace Rcpp;

const int MAX1 = 120;  // number of instances for train data
const int MAX2 = 30;   // number of instances for test data

//-----------------------------------

//A struct to store information about each observation
struct observation
{
  double sLength;      // Sepal.Length
  double sWidth;       // Sepal.Width
  double pLength;      // Petal.Length
  double pWidth;       // Petal.Width
  double distance;     // distance from the test point for this observation
  string species;      // type of species
};

//-----------------------------------------------------------------------
// comparison function for sorting
//----------------------------------------------------------------------
bool comparison(observation a, observation b) 
{ 
  return (a.distance < b.distance); 
} 

//-----------------------------------------------------------------------
// function to classify each test observation
//----------------------------------------------------------------------

string classifySpecies (vector<observation> train, observation test, int k)
{
  int i;
  for (i = 0; i < train.size() ; i++)
  {
    // using  Euclidean Distance, calculate the distance for each observation from observation 'test'
    train[i].distance = sqrt((long double)(((train[i].sLength - test.sLength) * (train[i].sLength - test.sLength)) 
                                         + ((train[i].sWidth - test.sWidth) * (train[i].sWidth - test.sWidth)) 
                                         + ((train[i].pLength - test.pLength) * (train[i].pLength - test.pLength)) 
                                         + ((train[i].pWidth - test.pWidth) * (train[i].pWidth - test.pWidth))));
  }
  
  sort(train.begin(),train.end(), comparison);  // sorts the training observations based on comparing the distances from the test observation
  
  int setosa = 0;
  int versicolor = 0;
  int virginica = 0;
  string type;
  
  // find the majority of the species from the neighbors
  for (i = 0; i < k ; i++)
  {
    if (train[i].species[0] == 's')
      setosa++;
    else if (train[i].species[1] == 'e')
      versicolor++;
    else
      virginica++;
  }
  
  // assign the type to the observation 'test'
  if ((setosa > versicolor) && (setosa > virginica))
    type = "setosa" ;       //setosa
  else if ((versicolor > setosa) && (versicolor > virginica))
    type = "versicolor";    //versicolor
  else
    type = "virginica";     //virginica
  return type;
}

//-----------------------------------------------------------------------
// Function to read vectors from given file
//----------------------------------------------------------------------

void readData (string fileName, vector<observation> & p)
{
  int i = 0;
  string temp;
  ifstream file(fileName.c_str());
  
  if (file.is_open())
  {
    // If file was opened successfully,
    getline(file, temp);
    
    while (getline(file, temp, ','))
    {
      p[i].sLength = stof(temp.c_str()); 
      getline(file, temp, ',');
      p[i].sWidth = stof(temp.c_str());
      getline(file, temp, ',');
      p[i].pLength = stof(temp.c_str());
      getline(file, temp, ',');
      p[i].pWidth = stof(temp.c_str());  
      getline(file, temp, '\n');
      p[i].species = temp;
      p[i].species.erase(remove( p[i].species.begin(), p[i].species.end(), '\"' ), p[i].species.end()); // erases double quotes from the string
      i++;
    }
  }
  else
    //If file was not opened, displays the error message
    cout << "FILE NOT OPENED!";
}



//---------------------------------------------------------------------------------------
// Function to classify all the observations in test data and print the required metrics
//---------------------------------------------------------------------------------------

void classify(vector<observation> train, vector<observation> test)
{
  int confusionMatrix[4][4];
  
  int i;
  int j;
  int counter = 0;
  string result[MAX2];
  for ( i = 0; i < test.size(); i++ )
  {
    result[i] = classifySpecies(train,test[i],3);    // classifies the test observation
    if(result[i].compare(test[i].species) == 0)      // checks if the results match the actual value
      counter++;
         
  }
  cout << endl;
  long double accuracy = ((double)counter / (double)test.size()) * 100;
  cout << "The accuracy is " << accuracy << "%" << endl << endl;
  
  //---------------------- For CONFUSION MATRIX -------------------------------------------
  
  //initialize the matrix
  for (i = 1; i < 4 ; i++)
    for(j = 1; j < 4 ; j++)
      confusionMatrix[i][j] = 0;
  
  //for column 1,
  int countSetosa = 0;
  int countVersicolor = 0;
  int countVirginica = 0;
  for (j = 0; j < test.size() ; j++ )
    {
      if ((result[j].compare("setosa") == 0) && (result[j].compare(test[j].species) == 0))
        countSetosa++;
      else if ((result[j].compare("setosa") == 0) && (test[j].species.compare("versicolor") == 0))
        countVersicolor++;
      else if ((result[j].compare("setosa") == 0) && (test[j].species.compare("virginica") == 0))
        countVirginica++;
      else;
    }
  confusionMatrix[1][1] = countSetosa;
  confusionMatrix[2][1] = countVersicolor;
  confusionMatrix[3][1] = countVirginica;
  
  //for column 2,
  countSetosa = 0;
  countVersicolor = 0;
  countVirginica = 0;
  for (j = 0; j < test.size() ; j++ )
  {
    if ((result[j].compare("versicolor") == 0) && (result[j].compare(test[j].species) == 0))
      countVersicolor++;
    else if ((result[j].compare("versicolor") == 0) && (test[j].species.compare("setosa") == 0))
      countSetosa++;
    else if ((result[j].compare("versicolor") == 0) && (test[j].species.compare("virginica") == 0))
      countVirginica++;
    else;
  }
  confusionMatrix[1][2] = countSetosa;
  confusionMatrix[2][2] = countVersicolor;
  confusionMatrix[3][2] = countVirginica;
  
  //for column 3,
  countSetosa = 0;
  countVersicolor = 0;
  countVirginica = 0;
  for (j = 0; j < test.size() ; j++ )
  {
    if ((result[j].compare("virginica") == 0) && (result[j].compare(test[j].species) == 0))
      countVirginica++;
    else if ((result[j].compare("virginica") == 0) && (test[j].species.compare("setosa") == 0))
      countSetosa++;
    else if ((result[j].compare("virginica") == 0) && (test[j].species.compare("versicolor") == 0))
      countVersicolor++;
    else;
  }
  confusionMatrix[1][3] = countSetosa;
  confusionMatrix[2][3] = countVersicolor;
  confusionMatrix[3][3] = countVirginica;
  
  // Prints the confusion matrix
  
  string types[3];
  types[0] = "setosa";
  types[1] = "versicolor";
  types[2] = "virginica";
  int m = 0;
  bool flag = true;
  cout << "CONFUSION MATRIX: " << endl;
  cout << "                    Predicted Labels" << endl;
  cout << "Test Labels \tsetosa \t versicolor \t virginica" << endl;
  for (i = 1; i < 4 ; i++)
  {
    if (flag)
    {
      cout << types[m++] << "\t\t";
      flag = false;
    }
    else
      cout << types[m++] << "\t";
    for(j = 1; j < 4 ; j++)
    {
      cout << confusionMatrix[i][j] << "\t\t";
    }
    cout << endl;
  }
}

// [[Rcpp::export]]
int main()
{
  // start time is recorded
  chrono::steady_clock::time_point begin = chrono::high_resolution_clock::now(); 
  
  vector<observation> train;
  train.resize(MAX1);
  readData("train.csv", train);  // reads the train observations
  
  vector<observation> test;
  test.resize(MAX2);
  readData("test.csv", test);    // reads the test observations
  
  classify(train, test);        // implements the kNN algorithm
  
  // end time is recorded
  chrono::steady_clock::time_point end = chrono::high_resolution_clock::now();    
  chrono::duration<double> dur = end - begin;   // difference is calculated
  cout << "The run time for cpp file is " << dur.count() * 1000 << "ms" << endl;
  
}

// Displays the results of the algorithm in the R console

/*** R
main()
*/
