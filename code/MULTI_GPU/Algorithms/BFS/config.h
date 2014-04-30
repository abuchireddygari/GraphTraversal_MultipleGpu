/**
Copyright 2013-2014 SYSTAP, LLC.  http://www.systap.com

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This work was (partially) funded by the DARPA XDATA program under
AFRL Contract #FA8750-13-C-0002.

This material is based upon work supported by the Defense Advanced
Research Projects Agency (DARPA) under Contract No. D14PC00029.
*/

#pragma once

#include <map>
#include <string>
#include <string.h>  //strtok
#include <typeinfo>
#include <iostream>

/******************************************
 * A class for storing a typeless parameter
 *****************************************/
class Parameter {
  public:
    Parameter() { memset(data,0,100); }
    template<typename T> Parameter(T value) { 
      set(value); 
    }
    template<typename T> T get() { 
      T value=*reinterpret_cast<T*>(&data[0]);
      return value;

    } //return the parameter as the templated type
    template<typename T> void set(T value) { *reinterpret_cast<T*>(&data[0])=value; }  //set the parameter from the templated type

  private:
    char data[100]; //8 bytes of storage
};

/*******************************************
 * A class to store a description of a 
 * parameter
 ******************************************/
class ParameterDescription {
  public:
    ParameterDescription() : type(0) {}
    ParameterDescription(const ParameterDescription &p) : type(p.type), name(p.name), description(p.description), default_value(default_value) {}
    ParameterDescription(const std::type_info *type, const std::string &name, const std::string &description, const Parameter& default_value) : type(type), name(name), description(description), default_value(default_value) {}
    const std::type_info *type;   //the type of the parameter
    std::string name;             //the name of the parameter
    std::string description;      //description of the parameter
    Parameter default_value;      //the default value of the parameter
};

/***********************************************
 * A class for storing paramaters in a database
 * which includes type information. 
 **********************************************/
class Config {
  public:
    Config();
    /***********************************************
     * Registers the parameter in the database.
    **********************************************/
    template <typename Type> static void registerParameter(std::string name, std::string description, Type default_value);
    
    /********************************************
    * Gets a parameter from the database and
    * throws an exception if it does not exist.
    *********************************************/
    template <typename Type>
    Type getParameter(std::string name)
    {
    	 //verify the paramter has been registered
    	  ParamDesc::iterator desc_iter=param_desc.find(name);
    	  if(desc_iter==param_desc.end()) {
    	    std::cout << "getParameter error: '" << name << "' not found\n";
    	    throw;
    	  }

//    	  std::cout << "string type: "; //<< typeid(std::string).name() << std::endl;

    	  //verify the types match
    	  if(desc_iter->second.type!=&typeid(Type))
    	  {
    		  std::cout << desc_iter->second.type->name() << std::endl;
    		  			std::cout << typeid(Type).name() << std::endl;
    	    std::cout << "getParameter error: '" << name << "' type miss match\n";
    	    throw;
    	  }

    	  //check if the paramter has been set
    	  ParamDB::iterator param_iter=params.find(name);
    	  if(param_iter==params.end()) {
    	    return desc_iter->second.default_value.get<Type>(); //return the default value
    	  }
    	  else {
    	    return param_iter->second.get<Type>();              //return the parameter value
    	  }
    }
    
    /**********************************************
    * Sets a parameter in the database
    * throws an exception if it does not exist.
    *********************************************/
    template <typename Type> void setParameter(std::string name, Type value)
    {
    	  //verify that the parameter has been registered
    	  ParamDesc::iterator iter=param_desc.find(name);
    	  if(iter==param_desc.end()) {
    	    std::cout << "setParameter error: '" << name << "' not found\n";
    	    throw;
    	  }
    	  if(iter->second.type!=&typeid(Type)) {
    	    std::cout << "setParameter error: '" << name << "' type miss match\n";
    	    throw;
    	  }
    	  params[name]=value;
    }
    
    /****************************************************
    * Parse paramters in the format 
    * name=value name=value ... name=value
    * and store the variables in the parameter database
    ****************************************************/
    void parseParameterString(char* str);

    /****************************************************
    * Parse a config file  in the format 
    * name=value
    * name=value
    * ...
    * name=value
    * and store the variables in the parameter database
    ****************************************************/
    void parseFile(const char* filename);

    /****************************************************
     * Print the options
     ***************************************************/
    static void printOptions();

    /***************************************************
     * Prints parameters                       *
     ***************************************************/
    void printConfig();

  private:
    typedef std::map<std::string,ParameterDescription> ParamDesc;
    typedef std::map<std::string,Parameter> ParamDB;
    
    static ParamDesc param_desc;  //The parameter descriptions
    ParamDB params;               //The parameter database

    /****************************************************
    * Parse a string in the format 
    * name=value
    * and store the variable in the parameter database
    ****************************************************/
    void setParameter(const char* str);

//    static SignalHandler sh;  //install the signal handlers here
    static bool registered;
};
