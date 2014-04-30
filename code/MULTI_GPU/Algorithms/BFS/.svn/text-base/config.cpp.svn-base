#include<config.h>
#include <iostream>
#include <getvalue.h>

Config::ParamDesc Config::param_desc;
bool Config::registered=false;

void Config::parseParameterString(char* str) {
  //copy to a temperary array to avoid destroying the string
  char params[1000]; 
  strncpy(params,str,1000);

  //tokenize
  char *var=strtok(params," ,");
  while(var!=0)
  {
    setParameter(var);
    var=strtok(NULL," ");
  }
}

#include<fstream>
void Config::parseFile(const char* filename) {
  std::ifstream fin;
  fin.open(filename);
  if(!fin)  
  {
//    char error[500];
//    sprintf(error,"Error opening file '%s'",filename);
    printf("Error opening file '%s'", filename);
  }

  while(!fin.eof())
  {
    char line[1000];
    fin.getline(line,1000,'\n');
    parseParameterString(line);
  }
  fin.close();
}

template <typename Type> 
void Config::registerParameter(std::string name, std::string description, Type default_value) {
  param_desc[name]=ParameterDescription(&typeid(Type),name,description,default_value);
}

void Config::printOptions() {
  for(ParamDesc::iterator iter=param_desc.begin();iter!=param_desc.end();iter++)
  {
    std::cout << "           " << iter->second.name << ": " << iter->second.description << std::endl;
  }
}

void Config::setParameter(const char* str) {

  std::string tmp(str);

  //locate the split
  int split_loc=tmp.find("=");

  std::string name=tmp.substr(0,split_loc);
  std::string value=tmp.substr(split_loc+1);

  //verify parameter was registered
  ParamDesc::iterator iter=param_desc.find(std::string(name));
  if(iter==param_desc.end()) {
//    char error[100];
    printf("Variable '%s' not registered",name.c_str());
//    FatalError(error);
  }

  if(*(iter->second.type)==typeid(int))
    setParameter(name, getValue<int>(value.c_str()));
  else if(*(iter->second.type)==typeid(float))
    setParameter(name, getValue<float>(value.c_str()));
  else if(*(iter->second.type)==typeid(double))
    setParameter(name, getValue<double>(value.c_str()));
  else if(*(iter->second.type)==typeid(bool))
      setParameter(name, getValue<bool>(value.c_str()));
  else
  {
//    char error[100];
    printf("getValue is not implemented for the datatype of variable '%s'",name.c_str());
  }
}

#include <register.h>
Config::Config() {
  if(!registered) {
    registerParameters();
    registered=true;
  }
} 
