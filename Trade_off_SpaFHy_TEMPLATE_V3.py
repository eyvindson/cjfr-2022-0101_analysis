#   !/usr/bin/env python
# coding: utf-8

#DATA IMPORTATION
from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition
import pandas
import time
import numpy
import pandas as pd
import numpy as np
import pyutilib.services
import pickle
import random
import copy
import matplotlib.pyplot as plt
import statistics as stat

#path = "/home/ubuntu/workspace/pyomo/"
path = "/scratch/project_2000611/KYLE/AVO2/Files_for_optimization/"
pyutilib.services.TempfileManager.tempdir = path

path = "/scratch/project_2000611/KYLE/SpaFHy_manuscript/simulated_data/"
path_output = "/scratch/project_2000611/KYLE/SpaFHy_manuscript/"


if __name__ == '__main__':

    import argparse
    import multiprocessing as mp
    import glob
    import pickle
    parser = argparse.ArgumentParser()
    parser.add_argument('--area', help='which region, Keski-Suomi, Pohjois-Pohjanmaa, Uusimaa ', type=str)
    parser.add_argument('--trade', help ='Is the tradeoff between INC_PEAT, NPV_PEAT, or PEAT_NPV', type=str)
    parser.add_argument('--constraint', help ='Is the constraint INC, PEAT or NPV', type=str)
    
    args = parser.parse_args()
    
    RG = args.area #"Keski-Suomi"
    trade_off = args.trade #"INC_PEAT"
    constraint = args.constraint #"NPV"


    class optimization:
        def __init__(self):
            small = False
            
            #NO ditching
            self.data_no = pd.read_csv(path + "rslt_GWT_w_SPAFHY_PEAT_"+RG+"_no.csv")
            #Yes to ditching
            self.data_yes = pd.read_csv(path + "rslt_GWT_w_SPAFHY_PEAT_"+RG+"_yes.csv")
            #self.data_no.dropna(inplace = True)
            #self.data_yes.dropna(inplace = True)
            self.data_no = self.data_no[self.data_no['standid']>0]  
            self.data_yes = self.data_yes[self.data_yes['standid']>0]
            
            self.data = pd.concat([self.data_no,self.data_yes],axis=0)
            self.data = self.data.reset_index().drop(['index'],axis=1)
            
            self.dd1 = self.data[(~self.data['branching_group'].str.contains("Selection"))]
            self.dd2 = self.data[(self.data['branching_group'].str.contains("Selection")) & (self.data['MAINT']!="MAINTENANCE")]
            self.data = pd.concat([self.dd1,self.dd2])
            self.data = self.data[self.data['PEAT'] ==1]
            
            self.data['branching_group'] = self.data['branching_group']+self.data["branch_desc"]
            self.data['branching_group'] = self.data['branching_group'].str.replace("+","")
            self.data['branching_group'] = self.data['branching_group'].str.replace("-","m")
            self.data['branching_group'] = self.data['branching_group'].str.replace(" ","")
            self.data['branching_group'] = self.data['branching_group'].str.replace("_","")
            self.data['branching_group'] = self.data['branching_group'].str.replace("_","")
            self.data['branching_group'] = self.data['branching_group'].str.replace("|","")
            self.data['branching_group'].replace({"00": "SA"}, inplace=True)
            
            self.data = self.data[self.data["branching_group"]!= "00MAINTENANCE"]
            
            #If a subset is desired, change "/1" to "/10"
            n = int(len(set(self.data["id"].values))/1)
            self.stand_sample = list(set(self.data["id"].values))[0:int(n/1)]
            self.data = self.data[self.data["id"].isin(self.stand_sample)]
            self.data["ALT"] = ["SC" if x[0:4] == "Sele" else "FC" if x[0:4] == "Norm" else "SA" for x in self.data['branching_group']]
            self.data['cost'] = self.data['cash_flow'].where(self.data['cash_flow'] < 0, other=0)
            
            self.data.set_index(["id","branching_group","year"],inplace=True)
            self.options = list(set(self.data.index.get_level_values("branching_group")))
            self.stands = list(set(self.data.index.get_level_values("id")))
            self.years = sorted(list(set(self.data.index.get_level_values("year"))))
            
            self.AREA = self.data.loc[(slice(None),slice(None),self.years[0]),"AREA"]
            self.AREA = self.AREA.reset_index()
            self.AREA.drop(["branching_group","year"],axis=1,inplace=True)
            self.AREA.drop_duplicates(inplace=True)
            self.AREA.set_index("id",inplace=True)
            self.AREA.sort_index(inplace=True)
            
            self.AREA_ALL = self.data.loc[(slice(None),slice(None),slice(None)),"AREA"]
                   
            self.Harvested_V = self.data.loc[(slice(None),slice(None),slice(None)),"Harvested_V"]*self.AREA_ALL
            self.COSTS = self.data.loc[(slice(None),slice(None),slice(None)),"cost"]*self.AREA_ALL
            self.DEVEL_CLASS = self.data.loc[(slice(None),slice(None),slice(None)),"DEVEL_CLASS"]
            self.PV = self.data.loc[(slice(None),slice(None),slice(None)),"PV"]*self.AREA_ALL
            self.Carbon_soil = self.data.loc[(slice(None),slice(None),slice(None)),"Carbon_soil"]*self.AREA_ALL
            self.BM_total = self.data.loc[(slice(None),slice(None),slice(None)),"BM_total"]*self.AREA_ALL
            self.N2O = self.data.loc[(slice(None),slice(None),slice(None)),"N2O2010"]*self.AREA_ALL*10*5#Changing from g/m2 to kg /ha, and from single year to period (*5)
            self.CO2 = self.data.loc[(slice(None),slice(None),slice(None)),"CO22010"]*self.AREA_ALL*10*5
            self.CH4 = self.data.loc[(slice(None),slice(None),slice(None)),"CH42010"]*self.AREA_ALL*10*5
            
            self.N2O_MEAN =self.data.loc[(slice(None),slice(None),slice(None)),["N2O"+str(i) for i in range(2000,2016)]].mean(axis=1)*self.AREA_ALL*10*5
            self.CH4_MEAN =self.data.loc[(slice(None),slice(None),slice(None)),["CH4"+str(i) for i in range(2000,2016)]].mean(axis=1)*self.AREA_ALL*10*5
            self.CO2_MEAN =self.data.loc[(slice(None),slice(None),slice(None)),["CO2"+str(i) for i in range(2000,2016)]].mean(axis=1)*self.AREA_ALL*10*5
            
            self.Harvested_V_log = self.data.loc[(slice(None),slice(None),slice(None)),"income_log_change"]*self.AREA_ALL
            self.Harvested_V_pulp = self.data.loc[(slice(None),slice(None),slice(None)),"income_pulp_change"]*self.AREA_ALL
            
            self.H_V_LOG = self.data.loc[(slice(None),slice(None),slice(None)),"Harvested_V_log"]*self.AREA_ALL
            self.H_V_PULP = self.data.loc[(slice(None),slice(None),slice(None)),"Harvested_V_pulp"]*self.AREA_ALL
            self.H_V = self.data.loc[(slice(None),slice(None),slice(None)),"Harvested_V"]*self.AREA_ALL
            self.PEAT = self.data.loc[(slice(None),slice(None),slice(None)),"PEAT"]
            self.CLEARCUT = self.data.loc[(slice(None),slice(None),slice(None)),"clearcut"]
            self.data['MAINT'] = [1 if i[-11:] == "MAINTENANCE" else 0 for i in self.data.branch_desc]
            self.data["DIT_MAINT"] = [1 if i == 1 else 1 if i == 1 else 0 for i in self.data.SINCE_DRAINAGE]#**
            self.DITCHMAINT = self.data.loc[(slice(None),slice(None),slice(None)),"DIT_MAINT"]*500*self.AREA_ALL*self.data.MAINT
           
            self.infinity = float('inf')
            self.regime_set = list(self.options)
            self.stand_set = list(self.stands)
            self.stand_set = list()
            self.n_stands = len(self.stands)
            self.n_regimes = len(self.options)
            
            self.createModel()
        
        def createModel(self):
            
            # Declare sets
            self.model1 = ConcreteModel()
            
            self.model1.regimes = Set(initialize = self.regime_set)
            self.model1.periods = Set(initialize=self.years, doc='Years')
            
            self.t = self.Harvested_V
            self.t = self.t.reset_index()
            self.t1 = self.t[['id', 'branching_group']]
            self.t1 = self.t1.drop_duplicates()
            self.t2 = self.t[['id']]
            self.t2 = self.t2.drop_duplicates()
            self.t1["v"] = 0
            self.t1.set_index(["id","branching_group"],inplace=True)
            
            self.tSC = self.data.loc[(slice(None),slice(None),slice(None)),["ALT","PEAT"]]
            self.tSC = self.tSC.reset_index()
            self.tSCa = self.tSC[(self.tSC["ALT"] != "FC") & (self.tSC["PEAT"] == 1)]
            self.tSCb = self.tSC[(self.tSC["PEAT"] == 0)]
            self.tSC = self.tSCa.append(self.tSCb)
            self.t1SC = self.tSC[['id', 'branching_group']]
            self.t1SC = self.t1SC.drop_duplicates()
            self.t2SC = self.tSC[['id']]
            self.t2SC = self.t2SC.drop_duplicates()
            self.t1SC["v"] = 0
            self.t1SC.set_index(["id","branching_group"],inplace=True)

            self.tFC = self.data.loc[(slice(None),slice(None),slice(None)),["ALT","PEAT"]]
            self.tFC = self.tFC.reset_index()
            self.tFCa = self.tFC[(self.tFC["ALT"] != "SC") & (self.tFC["PEAT"] == 1)]
            self.tFCb = self.tFC[(self.tFC["PEAT"] == 0)]
            self.tFC = self.tFCa.append(self.tFCb)
            self.t1FC = self.tFC[['id', 'branching_group']]
            self.t1FC = self.t1FC.drop_duplicates()
            self.t2FC = self.tFC[['id']]
            self.t2FC = self.t2FC.drop_duplicates()
            self.t1FC["v"] = 0
            self.t1FC.set_index(["id","branching_group"],inplace=True)
            
            self.model1.stands = Set(initialize = list(self.t2['id']))

            # Indexes (stand, regime)
            def index_rule(model1):
                index = []
                for i in self.t1.index:
                    index.append((i[0],i[1]))
                return index
            self.model1.index1 = Set(dimen=2, initialize=index_rule)
            
            def index_rule_FC(model1):
                index = []
                for i in self.t1FC.index:
                    index.append((i[0],i[1]))
                return index
            self.model1.index1FC = Set(dimen=2, initialize=index_rule_FC)
            
            def index_rule_SC(model1):
                index = []
                for i in self.t1SC.index:
                    index.append((i[0],i[1]))
                return index
            self.model1.index1SC = Set(dimen=2, initialize=index_rule_SC)
            
            # Variables
            self.model1.X1 = Var(self.model1.index1, within=NonNegativeReals, bounds=(0,1), initialize=1)
            
            # Define the objective
            def outcome_rule(model1):
                return sum(self.Harvested_V.loc[s,r,self.years[0]] * model1.X1[(s,r)] for (s,r) in model1.index1)
            self.model1.OBJ = Objective(rule=outcome_rule, sense=maximize)
            
            def regime_rule(model1, t):
                row_sum = sum(model1.X1[(t,p)] for p in [x[1] for x in model1.index1 if x[0] == t])
                return row_sum == 1
            self.model1.regime_limit = Constraint(self.model1.stands, rule=regime_rule)
        
        def solve(self):
            opt = SolverFactory('cbc')
            self.results = opt.solve(self.model1,tee=False)

    def CALC_MIN_MAX(t3,t3str,name,DATA2):
        f = open(path + "FOR_r_"+name+".txt", "w")
        MF_var = ["N2O","CO2","CH4","Carbon_soil","BM_total","Harvested_V"]
        
        for VAR_i in MF_var:
            t3.model1.del_component(t3.model1.OBJ)    
            # Define the objective
            if VAR_i in ["N2O","CO2","CH4"]:
                def outcome_rule(model1):
                    value = str("sum("+str(t3str)+"."+VAR_i+"_MEAN.loc[s,r,year] * "+str(t3str)+".model1.X1[(s,r)] * "+str(t3str) + ".PEAT.loc[s,r,year] for (s,r) in "+str(t3str)+".model1.index1 for year in "+str(t3str)+".years)")
                    setattr(t3,"out",eval(value))
                    return t3.out
            elif VAR_i in ["Carbon_soil"]:
                def outcome_rule(model1):
                    value = str("sum("+str(t3str)+"."+VAR_i+".loc[s,r,max(t3.years)] * "+str(t3str)+".model1.X1[(s,r)] * (1-"+str(t3str) + ".PEAT.loc[s,r,max(t3.years)]) for (s,r) in "+str(t3str)+".model1.index1)")
                    setattr(t3,"out",eval(value))
                    return t3.out
            elif VAR_i in ["BM_total"]:
                def outcome_rule(model1):
                    value = str("sum("+str(t3str)+"."+VAR_i+".loc[s,r,max(t3.years)] * "+str(t3str)+".model1.X1[(s,r)] for (s,r) in "+str(t3str)+".model1.index1)")
                    setattr(t3,"out",eval(value))
                    return t3.out
            else:
                def outcome_rule(model1):
                    value = value = str("sum("+str(t3str)+"."+VAR_i+".loc[s,r,year] * "+str(t3str)+".model1.X1[(s,r)] for (s,r) in "+str(t3str)+".model1.index1 for year in "+str(t3str)+".years)")
                    setattr(t3,"out",eval(value))
                    return t3.out
            t3.model1.OBJ = Objective(rule=outcome_rule, sense=maximize)
            t3.solve()
            DATA1 = DATA2
            b1 = pd.DataFrame(t3.model1.X1)
            b1["Dec"] = 0.0
            for l in range(0,len(b1)):
                b1["Dec"][l] = t3.model1.X1[(b1[0][l],b1[1][l])].value
            b1 = b1.rename(columns = {0:"id",1:"branching_group","Dec":"DEC"})
            b1.set_index(["id","branching_group"],inplace=True)
            result = DATA1.join(b1, how='inner')
            result_P10 = result.loc[(result.index.get_level_values('year') == max(t3.years))]
            if VAR_i in ["N2O","CO2","CH4"]:  
                value_VAR_i = "sum((result['"+VAR_i+"_MEAN']*result['PEAT']) * result['DEC'])"
            elif VAR_i in ["Carbon_soil"]:
                value_VAR_i = "sum((result_P10['"+VAR_i+"']*(1-result_P10['PEAT'])) * result_P10['DEC'])"
            elif VAR_i in ["BM_total"]:
                value_VAR_i = "sum((result_P10['"+VAR_i+"']) * result_P10['DEC'])"
            else:
                value_VAR_i = "sum((result['"+VAR_i+"']) * result['DEC'])"
            setattr(t3,"max_"+VAR_i, eval(value_VAR_i ))
            t3.model1.del_component(t3.model1.OBJ)    
            t3.model1.OBJ = Objective(rule=outcome_rule, sense=minimize)
            t3.solve()
            DATA1 = DATA2
            b1 = pd.DataFrame(t3.model1.X1)
            b1["Dec"] = 0
            for l in range(0,len(b1)):
                b1["Dec"][l] = t3.model1.X1[(b1[0][l],b1[1][l])].value
            b1 = b1.rename(columns = {0:"id",1:"branching_group","Dec":"DEC"})
            b1.set_index(["id","branching_group"],inplace=True)
            result = DATA1.join(b1, how='inner')
            result_P10 = result.loc[(result.index.get_level_values('year') == max(t3.years))]
            if VAR_i in ["N2O","CO2","CH4"]:  
                value_VAR_i = "sum((result['"+VAR_i+"_MEAN']*result['PEAT']) * result['DEC'])"
            elif VAR_i in ["Carbon_soil"]:
                value_VAR_i = "sum((result_P10['"+VAR_i+"']*(1-result_P10['PEAT'])) * result_P10['DEC'])"
            elif VAR_i in ["BM_total"]:
                value_VAR_i = "sum((result_P10['"+VAR_i+"']) * result_P10['DEC'])"
            else:
                value_VAR_i = "sum((result['"+VAR_i+"']) * result['DEC'])"
            try:
                setattr(t3,"min_"+VAR_i, eval(value_VAR_i))
            except:
                print(value_VAR_i)
            print("mm"+str(VAR_i)+"<-c("+str(getattr(t3,"max_"+VAR_i))+","+str(getattr(t3,"min_"+VAR_i))+")")
            f.write("mm"+str(VAR_i)+"<-c("+str(getattr(t3,"max_"+VAR_i))+","+str(getattr(t3,"min_"+VAR_i))+")\r")
        f.close()
        d = {'min': [getattr(t3,"min_"+VAR_i) for VAR_i in MF_var], 'min': [getattr(t3,"max_"+VAR_i) for VAR_i in MF_var]}
        df = pd.DataFrame(data=d, index = MF_var)
        return df



        import argparse
        import multiprocessing as mp
        import glob
        import pickle
        parser = argparse.ArgumentParser()
        parser.add_argument('--weather', help='which weather region, lanssuomi, lappi, itasuomi... ', type=str)
        parser.add_argument('--database', help ='from SIMO', type=str)
        parser.add_argument('--region', help ='Name of region  -- Uusimaa, Pohjois-Pohjanmaa or Keski-Suomi', type=str)
        
        args = parser.parse_args()
        
        DATA_ANALYSIS(args.weather,args.database,args.region)
        



        import argparse
        import multiprocessing as mp
        import glob
        import pickle
        parser = argparse.ArgumentParser()
        parser.add_argument('--weather', help='which weather region, lanssuomi, lappi, itasuomi... ', type=str)
        parser.add_argument('--database', help ='from SIMO', type=str)
        parser.add_argument('--region', help ='Name of region  -- Uusimaa, Pohjois-Pohjanmaa or Keski-Suomi', type=str)
        
        args = parser.parse_args()
        
        DATA_ANALYSIS(args.weather,args.database,args.region)
        
    t2 = optimization()    

    t3 = copy.deepcopy(t2)
    str_t3 = "t3"
    calc_MAX = 0

    print(t3)
    print("Loaded problem")
    #CALCULATE MINIMUM AND MAXIMUM OF EACH CRITION WITHOUT CONSTRAINTS

    #This section of code allows for array multiplication -- faster than loops.
    DATA = t3.Harvested_V_log+t3.Harvested_V_pulp-t3.DITCHMAINT
    DATA = pd.DataFrame(DATA)
    DATA = DATA.rename(columns = {0:"Harvested_V"})
    peat = t3.PEAT
    peat = pd.DataFrame(peat)
    DATA["PEAT"]= peat["PEAT"]
    Harvested_V_log = t3.Harvested_V_log
    Harvested_V_log = pd.DataFrame(Harvested_V_log)
    DATA["Harvested_V_log"]= Harvested_V_log[0]
    Harvested_V_pulp = t3.Harvested_V_pulp
    Harvested_V_pulp = pd.DataFrame(Harvested_V_pulp)
    DATA["Harvested_V_pulp"]= Harvested_V_pulp[0]

    H_V_LOG = t3.H_V_LOG
    H_V_LOG = pd.DataFrame(H_V_LOG)
    DATA["H_V_LOG"]= H_V_LOG[0]
    H_V_PULP = t3.H_V_PULP
    H_V_PULP = pd.DataFrame(H_V_PULP)
    DATA["H_V_PULP"]= H_V_PULP[0]
    H_V = t3.H_V
    H_V = pd.DataFrame(H_V)
    DATA["H_V"]= H_V[0]

    DITCHMAINT = t3.DITCHMAINT
    DITCHMAINT = pd.DataFrame(DITCHMAINT)
    DATA["DITCHMAINT"] = DITCHMAINT[0]
    CLEAR = t3.CLEARCUT
    CLEAR = pd.DataFrame(CLEAR)
    DATA["CLEARCUT"]= CLEAR["clearcut"]
    PV = t3.PV
    PV = pd.DataFrame(PV)
    DATA["PV"]= PV[0]

    N2O = t3.N2O
    N2O = pd.DataFrame(N2O)
    DATA["N2O"]= N2O[0]
    CO2 = t3.CO2
    CO2 = pd.DataFrame(CO2)
    DATA["CO2"]= CO2[0]
    CH4 = t3.CH4
    CH4 = pd.DataFrame(CH4)
    DATA["CH4"]= CH4[0]
    Carbon_soil = t3.Carbon_soil
    Carbon_soil = pd.DataFrame(Carbon_soil)
    DATA["Carbon_soil"]= Carbon_soil[0]
    BM_total = t3.BM_total
    BM_total = pd.DataFrame(BM_total)
    DATA["BM_total"]= BM_total[0]
    AREA_ALL = t3.AREA_ALL
    AREA_ALL = pd.DataFrame(AREA_ALL)
    DATA["AREA_ALL"]= AREA_ALL["AREA"]

    DATA["DEVEL_CLASS"] = pd.DataFrame(t3.DEVEL_CLASS)

    CH4_MEAN = t3.CH4_MEAN
    CH4_MEAN = pd.DataFrame(CH4_MEAN)
    N2O_MEAN = t3.N2O_MEAN
    N2O_MEAN = pd.DataFrame(N2O_MEAN)
    CO2_MEAN = t3.CO2_MEAN
    CO2_MEAN = pd.DataFrame(CO2_MEAN)
    DATA["CO2_MEAN"] = CO2_MEAN[0]
    DATA["CH4_MEAN"] = CH4_MEAN[0]
    DATA["N2O_MEAN"] = N2O_MEAN[0]

    D1 = DATA
    D1 = D1.reset_index()
    D1['Y1']= D1['year']
    D1.set_index(["id","branching_group","year"],inplace=True)
    DATA['Y1']=D1['Y1']


    #CALCULATION MAX AND MIN

    min_max = CALC_MIN_MAX(t3,"t3","TEST",DATA)
    print("Calculated Max and Min values")
        
    kk = ["FC","SC","NONE"]
    kk = ["NONE"]
    #t2 = optimization()
    str_t3 = "t3"

    NO_CF_SC = [[0,0],[0,0],[0,0]]

    max_inc = []
    max_npv = []
    min_npv = []
        
    payoff_table = {}
    for k in kk:
        t3 = copy.deepcopy(t2)
        #Change objective to NPV
        t3.model1.del_component(t3.model1.OBJ)   
        #discount rate
        rr = 0.03
        
        def regime_rule_SC(model1, t):
            row_sum = sum(model1.X1[(t,p)] for p in [x[1] for x in model1.index1SC if x[0] == t])
            return row_sum == 1
        def regime_rule_FC(model1, t):
            row_sum = sum(model1.X1[(t,p)] for p in [x[1] for x in model1.index1FC if x[0] == t])
            return row_sum == 1
        
        if k == "SC":
            print("SELECTION HARVEST")
            t3.model1.regime_limit_SC = Constraint(t3.model1.stands, rule=regime_rule_SC)
        elif k == "FC":
            print("CLEAR FELLING")
            t3.model1.regime_limit_FC = Constraint(t3.model1.stands, rule=regime_rule_FC)
        else:
            print("NO RESTRICTION")
            
        t3.model1.DEC_inc = Var(within=NonNegativeReals)#, bounds=(0,1), initialize=1)
        
        def outcome_rule(model1):
            #INC  = sum((t3.Harvested_V_log.loc[s,t3.years[0],r]+t3.Harvested_V_pulp.loc[s,t3.years[0],r]) * t3.model1.X1[(s,r)] for (s,r) in t3.model1.index1)
            return t3.model1.DEC_inc
        t3.model1.OBJ = Objective(rule=outcome_rule, sense=maximize)
        
        def INC_bounded_rule(model1,year):
            #INC = sum((t3.Harvested_V_log.loc[s,t3.years[0],r]+t3.Harvested_V_pulp.loc[s,t3.years[0],r]) * t3.model1.X1[(s,r)] for (s,r) in t3.model1.index1)
            INC2 = (sum((t3.Harvested_V_log.loc[s,r,year]+t3.Harvested_V_pulp.loc[s,r,year]) * t3.model1.X1[(s,r)] for (s,r) in t3.model1.index1))
            return INC2 >= t3.model1.DEC_inc
        t3.model1.INC_bounded = Constraint(t3.years,rule=INC_bounded_rule)    
        
        t3.solve()
        EM_VAR = ["CH4","N2O","CO2"]
        BM = (sum((t3.BM_total.loc[s,r,max(t3.years)]-t3.BM_total.loc[s,'SA',max(t3.years)] )* t3.model1.X1[(s,r)].value for (s,r) in t3.model1.index1))
        BM_CO2EQV = BM*0.5*(44/12)
        CH4 = sum((t3.CH4_MEAN.loc[s,r,year]) * t3.model1.X1[(s,r)].value * t3.PEAT.loc[s,r,year] for (s,r) in t3.model1.index1 for year in t3.years)
        N2O = sum((t3.N2O_MEAN.loc[s,r,year]) * t3.model1.X1[(s,r)].value * t3.PEAT.loc[s,r,year] for (s,r) in t3.model1.index1 for year in t3.years)
        CO2 = sum((t3.CO2_MEAN.loc[s,r,year]) * t3.model1.X1[(s,r)].value * t3.PEAT.loc[s,r,year] for (s,r) in t3.model1.index1 for year in t3.years)
            
        CH4_CO2EQV = CH4*25
        N2O_CO2EQV = N2O*298
        max_PEAT  = (N2O_CO2EQV+CH4_CO2EQV+CO2)#+BM_CO2EQV
        
        NPV  = sum((t3.Harvested_V_log.loc[s,r,year]+t3.Harvested_V_pulp.loc[s,r,year]-t3.DITCHMAINT.loc[s,r,year]-t3.COSTS.loc[s,r,year])/((1+rr)**(2.5+year-2016)) * t3.model1.X1[(s,r)].value for (s,r) in t3.model1.index1 for year in t3.years)+sum(t3.PV.loc[s,r,max(t3.years)]/((1+rr)**(max(t3.years)-2016)) * t3.model1.X1[(s,r)].value for (s,r) in t3.model1.index1)
        
        payoff_table["1"] = [min([(sum((t3.Harvested_V_log.loc[s,r,year]+t3.Harvested_V_pulp.loc[s,r,year]) * t3.model1.X1[(s,r)].value for (s,r) in t3.model1.index1)) for year in t3.years]), max_PEAT, NPV]
        max_INC = t3.model1.DEC_inc.value#sum((t3.Harvested_V_log.loc[s,t3.years[0],r]+t3.Harvested_V_pulp.loc[s,t3.years[0],r]) * t3.model1.X1[(s,r)].value for (s,r) in t3.model1.index1)
        print("NEW")
        print(max_INC)
        
        
        max_inc = max_inc + [max_INC]
        
        t3.model1.del_component(t3.model1.INC_bounded)   
        t3.model1.del_component(t3.model1.INC_bounded_index)   
        
        def INC_bounded_rule(model1,year):
            #INC = sum((t3.Harvested_V_log.loc[s,t3.years[0],r]+t3.Harvested_V_pulp.loc[s,t3.years[0],r]) * t3.model1.X1[(s,r)] for (s,r) in t3.model1.index1)
            INC2 = (sum((t3.Harvested_V_log.loc[s,r,year]+t3.Harvested_V_pulp.loc[s,r,year]) * t3.model1.X1[(s,r)] for (s,r) in t3.model1.index1))
            return INC2 >= 0
        t3.model1.INC_bounded = Constraint(t3.years,rule=INC_bounded_rule)  
        
        t3.model1.del_component(t3.model1.OBJ)   
        
        def outcome_rule(model1):
            EM_VAR = ["CH4","N2O","CO2"]
            BM = eval("(sum((t3.BM_total.loc[s,r,max(t3.years)]-t3.BM_total.loc[s,'SA',max(t3.years)] )* t3.model1.X1[(s,r)] for (s,r) in t3.model1.index1))")
            BM_CO2EQV = BM*0.5*(44/12)
            CH4 = sum((t3.CH4_MEAN.loc[s,r,year]) * t3.model1.X1[(s,r)] * t3.PEAT.loc[s,r,year] for (s,r) in t3.model1.index1 for year in t3.years)
            N2O = sum((t3.N2O_MEAN.loc[s,r,year]) * t3.model1.X1[(s,r)] * t3.PEAT.loc[s,r,year] for (s,r) in t3.model1.index1 for year in t3.years)
            CO2 = sum((t3.CO2_MEAN.loc[s,r,year]) * t3.model1.X1[(s,r)] * t3.PEAT.loc[s,r,year] for (s,r) in t3.model1.index1 for year in t3.years)
            CH4_CO2EQV = CH4*25
            N2O_CO2EQV = N2O*298
            PEAT  = (N2O_CO2EQV+CH4_CO2EQV+CO2)#+BM_CO2EQV
            return PEAT
        t3.model1.OBJ = Objective(rule=outcome_rule, sense=maximize)
        print("MAX GHG")
        t3.solve()
        
        EM_VAR = ["CH4","N2O","CO2"]
        BM = (sum((t3.BM_total.loc[s,r,max(t3.years)]-t3.BM_total.loc[s,'SA',max(t3.years)] )* t3.model1.X1[(s,r)].value for (s,r) in t3.model1.index1))
        BM_CO2EQV = BM*0.5*(44/12)
        CH4 = sum((t3.CH4_MEAN.loc[s,r,year]) * t3.model1.X1[(s,r)].value * t3.PEAT.loc[s,r,year] for (s,r) in t3.model1.index1 for year in t3.years)
        N2O = sum((t3.N2O_MEAN.loc[s,r,year]) * t3.model1.X1[(s,r)].value * t3.PEAT.loc[s,r,year] for (s,r) in t3.model1.index1 for year in t3.years)
        CO2 = sum((t3.CO2_MEAN.loc[s,r,year]) * t3.model1.X1[(s,r)].value * t3.PEAT.loc[s,r,year] for (s,r) in t3.model1.index1 for year in t3.years)
            
        CH4_CO2EQV = CH4*25
        N2O_CO2EQV = N2O*298
        max_PEAT  = (N2O_CO2EQV+CH4_CO2EQV+CO2)#+BM_CO2EQV
        
        NPV  = sum((t3.Harvested_V_log.loc[s,r,year]+t3.Harvested_V_pulp.loc[s,r,year]-t3.DITCHMAINT.loc[s,r,year]-t3.COSTS.loc[s,r,year])/((1+rr)**(2.5+year-2016)) * t3.model1.X1[(s,r)].value for (s,r) in t3.model1.index1 for year in t3.years)+sum(t3.PV.loc[s,r,max(t3.years)]/((1+rr)**(max(t3.years)-2016)) * t3.model1.X1[(s,r)].value for (s,r) in t3.model1.index1)
        
        payoff_table["2"] = [min([(sum((t3.Harvested_V_log.loc[s,r,year]+t3.Harvested_V_pulp.loc[s,r,year]) * t3.model1.X1[(s,r)].value for (s,r) in t3.model1.index1)) for year in t3.years]), max_PEAT, NPV]
        t3.model1.del_component(t3.model1.OBJ)   
        
        def outcome_rule(model1):
            EM_VAR = ["CH4","N2O","CO2"]
            BM = eval("(sum((t3.BM_total.loc[s,r,max(t3.years)]-t3.BM_total.loc[s,'SA',max(t3.years)] )* t3.model1.X1[(s,r)] for (s,r) in t3.model1.index1))")
            BM_CO2EQV = BM*0.5*(44/12)
            CH4 = sum((t3.CH4_MEAN.loc[s,r,year]) * t3.model1.X1[(s,r)] * t3.PEAT.loc[s,r,year] for (s,r) in t3.model1.index1 for year in t3.years)
            N2O = sum((t3.N2O_MEAN.loc[s,r,year]) * t3.model1.X1[(s,r)] * t3.PEAT.loc[s,r,year] for (s,r) in t3.model1.index1 for year in t3.years)
            CO2 = sum((t3.CO2_MEAN.loc[s,r,year]) * t3.model1.X1[(s,r)] * t3.PEAT.loc[s,r,year] for (s,r) in t3.model1.index1 for year in t3.years)
            CH4_CO2EQV = CH4*25
            N2O_CO2EQV = N2O*298
            PEAT  = (N2O_CO2EQV+CH4_CO2EQV+CO2)#+BM_CO2EQV
            return PEAT
        t3.model1.OBJ = Objective(rule=outcome_rule, sense=minimize)
        print("Min GHG")
        t3.solve()
        
        EM_VAR = ["CH4","N2O","CO2"]
        
        BM = eval("(sum((t3.BM_total.loc[s,r,max(t3.years)]-t3.BM_total.loc[s,'SA',max(t3.years)] )* t3.model1.X1[(s,r)].value for (s,r) in t3.model1.index1))")
        BM_CO2EQV = BM*0.5*(44/12)
        CH4 = sum((t3.CH4_MEAN.loc[s,r,year]) * t3.model1.X1[(s,r)].value * t3.PEAT.loc[s,r,year] for (s,r) in t3.model1.index1 for year in t3.years)
        N2O = sum((t3.N2O_MEAN.loc[s,r,year]) * t3.model1.X1[(s,r)].value * t3.PEAT.loc[s,r,year] for (s,r) in t3.model1.index1 for year in t3.years)
        CO2 = sum((t3.CO2_MEAN.loc[s,r,year]) * t3.model1.X1[(s,r)].value * t3.PEAT.loc[s,r,year] for (s,r) in t3.model1.index1 for year in t3.years)
        CH4_CO2EQV = CH4*25
        N2O_CO2EQV = N2O*298
        max_PEAT  = (N2O_CO2EQV+CH4_CO2EQV+CO2)#+BM_CO2EQV
        
        NPV  = sum((t3.Harvested_V_log.loc[s,r,year]+t3.Harvested_V_pulp.loc[s,r,year]-t3.DITCHMAINT.loc[s,r,year]-t3.COSTS.loc[s,r,year])/((1+rr)**(2.5+year-2016)) * t3.model1.X1[(s,r)].value for (s,r) in t3.model1.index1 for year in t3.years)+sum(t3.PV.loc[s,r,max(t3.years)]/((1+rr)**(max(t3.years)-2016)) * t3.model1.X1[(s,r)].value for (s,r) in t3.model1.index1)
        
        payoff_table["3"] = [min([(sum((t3.Harvested_V_log.loc[s,r,year]+t3.Harvested_V_pulp.loc[s,r,year]) * t3.model1.X1[(s,r)].value for (s,r) in t3.model1.index1)) for year in t3.years]), max_PEAT, NPV]    
        
        t3.model1.del_component(t3.model1.OBJ)   
        
        def outcome_rule(model1):
            NPV  = sum((t3.Harvested_V_log.loc[s,r,year]+t3.Harvested_V_pulp.loc[s,r,year]-t3.DITCHMAINT.loc[s,r,year]-t3.COSTS.loc[s,r,year])/((1+rr)**(2.5+year-2016)) * t3.model1.X1[(s,r)] for (s,r) in t3.model1.index1 for year in t3.years)+sum(t3.PV.loc[s,r,max(t3.years)]/((1+rr)**(max(t3.years)-2016)) * t3.model1.X1[(s,r)] for (s,r) in t3.model1.index1)
            return NPV
        t3.model1.OBJ = Objective(rule=outcome_rule, sense=maximize)
        print("MAX NPV")
        t3.solve()
        EM_VAR = ["CH4","N2O","CO2"]
        BM = eval("(sum((t3.BM_total.loc[s,r,max(t3.years)]-t3.BM_total.loc[s,'SA',max(t3.years)] )* t3.model1.X1[(s,r)].value for (s,r) in t3.model1.index1))")
        BM_CO2EQV = BM*0.5*(44/12)
        CH4 = sum((t3.CH4_MEAN.loc[s,r,year]) * t3.model1.X1[(s,r)].value * t3.PEAT.loc[s,r,year] for (s,r) in t3.model1.index1 for year in t3.years)
        N2O = sum((t3.N2O_MEAN.loc[s,r,year]) * t3.model1.X1[(s,r)].value * t3.PEAT.loc[s,r,year] for (s,r) in t3.model1.index1 for year in t3.years)
        CO2 = sum((t3.CO2_MEAN.loc[s,r,year]) * t3.model1.X1[(s,r)].value * t3.PEAT.loc[s,r,year] for (s,r) in t3.model1.index1 for year in t3.years)
            
        CH4_CO2EQV = CH4*25
        N2O_CO2EQV = N2O*298
        max_PEAT  = (N2O_CO2EQV+CH4_CO2EQV+CO2)#+BM_CO2EQV
        
        NPV  = sum((t3.Harvested_V_log.loc[s,r,year]+t3.Harvested_V_pulp.loc[s,r,year]-t3.DITCHMAINT.loc[s,r,year]-t3.COSTS.loc[s,r,year])/((1+rr)**(2.5+year-2016)) * t3.model1.X1[(s,r)].value for (s,r) in t3.model1.index1 for year in t3.years)+sum(t3.PV.loc[s,r,max(t3.years)]/((1+rr)**(max(t3.years)-2016)) * t3.model1.X1[(s,r)].value for (s,r) in t3.model1.index1)
        
        payoff_table["4"] = [min([(sum((t3.Harvested_V_log.loc[s,r,year]+t3.Harvested_V_pulp.loc[s,r,year]) * t3.model1.X1[(s,r)].value for (s,r) in t3.model1.index1)) for year in t3.years]), max_PEAT, NPV]
        
        NPV  = sum((t3.Harvested_V_log.loc[s,r,year]+t3.Harvested_V_pulp.loc[s,r,year]-t3.DITCHMAINT.loc[s,r,year]-t3.COSTS.loc[s,r,year])/((1+rr)**(2.5+year-2016)) * t3.model1.X1[(s,r)].value for (s,r) in t3.model1.index1 for year in t3.years)+sum(t3.PV.loc[s,r,max(t3.years)]/((1+rr)**(max(t3.years)-2016)) * t3.model1.X1[(s,r)].value for (s,r) in t3.model1.index1)
        
        max_NPV = sum((t3.Harvested_V_log.loc[s,r,year]+t3.Harvested_V_pulp.loc[s,r,year]-t3.DITCHMAINT.loc[s,r,year]-t3.COSTS.loc[s,r,year])/((1+rr)**(2.5+year-2016)) * t3.model1.X1[(s,r)].value for (s,r) in t3.model1.index1 for year in t3.years)+sum(t3.PV.loc[s,r,max(t3.years)]/((1+rr)**(max(t3.years)-2016)) * t3.model1.X1[(s,r)].value for (s,r) in t3.model1.index1)
        print(max_NPV)
        max_npv = max_npv + [max_NPV]
        
        t3.model1.del_component(t3.model1.OBJ)   
        
        def outcome_rule(model1):
            NPV  = sum((t3.Harvested_V_log.loc[s,r,year]+t3.Harvested_V_pulp.loc[s,r,year]-t3.DITCHMAINT.loc[s,r,year]-t3.COSTS.loc[s,r,year])/((1+rr)**(2.5+year-2016)) * t3.model1.X1[(s,r)] for (s,r) in t3.model1.index1 for year in t3.years)+sum(t3.PV.loc[s,r,max(t3.years)]/((1+rr)**(max(t3.years)-2016)) * t3.model1.X1[(s,r)] for (s,r) in t3.model1.index1)
            return NPV
        
        t3.model1.OBJ = Objective(rule=outcome_rule, sense=minimize)
        t3.solve()
        print("Min GHG")
        EM_VAR = ["CH4","N2O","CO2"]
        BM = eval("(sum((t3.BM_total.loc[s,r,max(t3.years)]-t3.BM_total.loc[s,'SA',max(t3.years)] )* t3.model1.X1[(s,r)].value for (s,r) in t3.model1.index1))")
        BM_CO2EQV = BM*0.5*(44/12)
        CH4 = sum((t3.CH4_MEAN.loc[s,r,year]) * t3.model1.X1[(s,r)].value * t3.PEAT.loc[s,r,year] for (s,r) in t3.model1.index1 for year in t3.years)
        N2O = sum((t3.N2O_MEAN.loc[s,r,year]) * t3.model1.X1[(s,r)].value * t3.PEAT.loc[s,r,year] for (s,r) in t3.model1.index1 for year in t3.years)
        CO2 = sum((t3.CO2_MEAN.loc[s,r,year]) * t3.model1.X1[(s,r)].value * t3.PEAT.loc[s,r,year] for (s,r) in t3.model1.index1 for year in t3.years)
        CH4_CO2EQV = CH4*25
        N2O_CO2EQV = N2O*298
        max_PEAT  = (N2O_CO2EQV+CH4_CO2EQV+CO2)#+BM_CO2EQV
        
        NPV  = sum((t3.Harvested_V_log.loc[s,r,year]+t3.Harvested_V_pulp.loc[s,r,year]-t3.DITCHMAINT.loc[s,r,year]-t3.COSTS.loc[s,r,year])/((1+rr)**(2.5+year-2016)) * t3.model1.X1[(s,r)].value for (s,r) in t3.model1.index1 for year in t3.years)+sum(t3.PV.loc[s,r,max(t3.years)]/((1+rr)**(max(t3.years)-2016)) * t3.model1.X1[(s,r)].value for (s,r) in t3.model1.index1)
        
        payoff_table["5"] = [min([(sum((t3.Harvested_V_log.loc[s,r,year]+t3.Harvested_V_pulp.loc[s,r,year]) * t3.model1.X1[(s,r)].value for (s,r) in t3.model1.index1)) for year in t3.years]), max_PEAT, NPV]
        NPV  = sum((t3.Harvested_V_log.loc[s,r,year]+t3.Harvested_V_pulp.loc[s,r,year]-t3.DITCHMAINT.loc[s,r,year]-t3.COSTS.loc[s,r,year])/((1+rr)**(2.5+year-2016)) * t3.model1.X1[(s,r)].value for (s,r) in t3.model1.index1 for year in t3.years)+sum(t3.PV.loc[s,r,max(t3.years)]/((1+rr)**(max(t3.years)-2016)) * t3.model1.X1[(s,r)].value for (s,r) in t3.model1.index1)
        
        min_NPV = sum((t3.Harvested_V_log.loc[s,r,year]+t3.Harvested_V_pulp.loc[s,r,year]-t3.DITCHMAINT.loc[s,r,year]-t3.COSTS.loc[s,r,year])/((1+rr)**(2.5+year-2016)) * t3.model1.X1[(s,r)].value for (s,r) in t3.model1.index1 for year in t3.years)+sum(t3.PV.loc[s,r,max(t3.years)]/((1+rr)**(max(t3.years)-2016)) * t3.model1.X1[(s,r)].value for (s,r) in t3.model1.index1)
        print(min_NPV)
        
        min_npv = min_npv + [min_NPV]

    range_inc_npv = {'min_NPV': payoff_table["5"][2],'max_NPV': payoff_table["4"][2],'max_INC': payoff_table["1"][0] ,'min_PEAT': payoff_table["4"][1],'max_PEAT': payoff_table["2"][1] }
    #range_inc_npv = {'min_NPV': min_npv,'max_NPV': max_npv,'max_INC': max_inc ,'min_PEAT': min_PEAT,'max_PEAT': max_PEAT }
    range_inc_npv = pd.DataFrame(data=range_inc_npv, index = kk)
    print(range_inc_npv)
    print(payoff_table)
    setattr(t3,"min_NPV", payoff_table["5"][2])
    setattr(t3,"max_INC", payoff_table["1"][0])
    setattr(t3,"max_PEAT", payoff_table["2"][1])
    setattr(t3,"min_PEAT", payoff_table["4"][1])
    setattr(t3,"max_NPV", payoff_table["4"][2])

    t3 = copy.deepcopy(t2)
    t3.model1.BM_CO2_EQV = Var(within=Reals)
    print("Creating new variables - for use in objective function")
    #EMISSIONS IN CO2 EQV.

    def objBM_CO2_EQV_rule(model1):
        BM = eval("(sum((t3.BM_total.loc[s,r,max(t3.years)]-t3.BM_total.loc[s,'SA',max(t3.years)] )* t3.model1.X1[(s,r)] for (s,r) in t3.model1.index1))")
        BM_CO2EQV = BM*0.5*(44/12)
        return t3.model1.BM_CO2_EQV == ((BM_CO2EQV))#-(t3.min_BM_total*0.5*(44/12)))/((t3.max_BM_total*0.5*(44/12))-(t3.min_BM_total*0.5*(44/12)))

    t3.model1.objBundle2 = Constraint(rule=objBM_CO2_EQV_rule)

    t3.model1.C_soil_CO2_EQV = Var(within=Reals)

    #EMISSIONS IN CO2 EQV.

    def objBM_C_soil_CO2_EQV_rule(model1):
        CS = eval("(sum((t3.Carbon_soil.loc[s,r,max(t3.years)]-t3.Carbon_soil.loc[s,'SA',max(t3.years)]) * t3.model1.X1[(s,r)] * (1-t3.PEAT.loc[s,r,max(t3.years)]) for (s,r) in t3.model1.index1))")
        CS_CO2EQV = CS*0.5*(44/12)
        return t3.model1.C_soil_CO2_EQV == ((CS_CO2EQV))#-(t3.min_Carbon_soil*0.5*(44/12)))/((t3.max_Carbon_soil*0.5*(44/12))-(t3.min_Carbon_soil*0.5*(44/12)))

    t3.model1.objBundle3 = Constraint(rule=objBM_C_soil_CO2_EQV_rule)

    t3.model1.PEAT_CO2_EQV = Var(within=Reals)

    EM_VAR = ["CH4","N2O","CO2"]
    #EMISSIONS IN CO2 EQV.

    def objEM_CO2_EQV_rule(model1):
        CH4 = sum((t3.CH4_MEAN.loc[s,r,year]) * t3.model1.X1[(s,r)] * t3.PEAT.loc[s,r,year] for (s,r) in t3.model1.index1 for year in t3.years)
        N2O = sum((t3.N2O_MEAN.loc[s,r,year]) * t3.model1.X1[(s,r)] * t3.PEAT.loc[s,r,year] for (s,r) in t3.model1.index1 for year in t3.years)
        CO2 = sum((t3.CO2_MEAN.loc[s,r,year]) * t3.model1.X1[(s,r)] * t3.PEAT.loc[s,r,year] for (s,r) in t3.model1.index1 for year in t3.years)        
        CH4_CO2EQV = CH4*25
        N2O_CO2EQV = N2O*298 
        return t3.model1.PEAT_CO2_EQV == ((CH4_CO2EQV + N2O_CO2EQV+CO2))

    t3.model1.objBundle4 = Constraint(rule=objEM_CO2_EQV_rule)

    t3.model1.PEAT_CO2_EQV_100 = Var(within=Reals)

    #EMISSIONS IN CO2 EQV.

    def objEM_CO2_EQV_rule_stoch(model1):
        CH4 = sum((t3.CH4_MEAN.loc[s,r,year]) * t3.model1.X1[(s,r)] * t3.PEAT.loc[s,r,year] for (s,r) in t3.model1.index1 for year in t3.years)
        N2O = sum((t3.N2O_MEAN.loc[s,r,year]) * t3.model1.X1[(s,r)] * t3.PEAT.loc[s,r,year] for (s,r) in t3.model1.index1 for year in t3.years)
        CO2 = sum((t3.CO2_MEAN.loc[s,r,year]) * t3.model1.X1[(s,r)] * t3.PEAT.loc[s,r,year] for (s,r) in t3.model1.index1 for year in t3.years)
        CH4_CO2EQV = CH4*25
        N2O_CO2EQV = N2O*298 
        return t3.model1.PEAT_CO2_EQV_100 == ((CH4_CO2EQV + N2O_CO2EQV+CO2))

    t3.model1.objBundle5 = Constraint(rule=objEM_CO2_EQV_rule_stoch)

    t3.model1.NPV_value = Var(within=Reals)

    #EMISSIONS IN CO2 EQV.

    def objEM_NPV_rule(model1):
        NPV  = sum((t3.Harvested_V_log.loc[s,r,year]+t3.Harvested_V_pulp.loc[s,r,year]-t3.DITCHMAINT.loc[s,r,year]-t3.COSTS.loc[s,r,year])/((1+rr)**(2.5+year-2016)) * t3.model1.X1[(s,r)] for (s,r) in t3.model1.index1 for year in t3.years)+sum(t3.PV.loc[s,r,max(t3.years)]/((1+rr)**(max(t3.years)-2016)) * t3.model1.X1[(s,r)] for (s,r) in t3.model1.index1)
        return t3.model1.NPV_value == NPV

    t3.model1.objEM_NPV_rule = Constraint(rule=objEM_NPV_rule)
    t3.model1.INC_value = Var(within=Reals)
    t3.model1.BM_CO2_EQV_100 = Var(within=Reals)

    #EMISSIONS IN CO2 EQV.

    def objBM_CO2_EQV_rule_stoch(model1):
        BM = eval("(sum([sum((t3.BM_total.loc[s,r,max(t3.years)]-t3.BM_total.loc[s,'SA',max(t3.years)] )* t3.model1.X1[(s,r)] for (s,r) in t3.model1.index1)]))")
        BM_CO2EQV_100 = BM*0.5*(44/12)
        return t3.model1.BM_CO2_EQV_100 == ((BM_CO2EQV_100))#-(t3.min_BM_total*0.5*(44/12)))/((t3.max_BM_total*0.5*(44/12))-(t3.min_BM_total*0.5*(44/12)))

    t3.model1.objBundle6 = Constraint(rule=objBM_CO2_EQV_rule_stoch)

    print("Created new variables")

    t3.model1.flow_p_inc = Param(default=1, mutable=True)
    t3.model1.max_INC_mut = Param(default=1, mutable=True)
    t3.model1.max_NPV_mut = Param(default=1, mutable=True)
    t3.model1.min_NPV_mut = Param(default=1, mutable=True)
    t3.model1.max_PEAT_mut = Param(default=1, mutable=True)
    t3.model1.min_PEAT_mut = Param(default=1, mutable=True)
    t3.model1.flow_p_OBJ = Param(default=1, mutable=True)
    t3.model1.DEC_inc = Var(within=NonNegativeReals, bounds=(0,1), initialize=1)

    def limit_regimes(t1,reg_type,OPT):
        
        def INC_bounded_rule(model1,year):
            INC2 = (sum((t1.Harvested_V_log.loc[s,r,year]+t1.Harvested_V_pulp.loc[s,r,year]) * t1.model1.X1[(s,r)] for (s,r) in t1.model1.index1)/t1.model1.max_INC_mut)
            return INC2 >= t1.model1.DEC_inc
        
        def INC_bound_lower_rule(model1,year):
            INC = (t1.model1.max_INC_mut*t1.model1.flow_p_inc)
            INC2 = sum((t1.Harvested_V_log.loc[s,r,year]+t1.Harvested_V_pulp.loc[s,r,year]) * t1.model1.X1[(s,r)] for (s,r) in t1.model1.index1)
            return INC2 >= INC
        
        t1.model1.flow_p_npv = Param(default=1, mutable=True)
        
        def NPV_bound_lower_rule(model1):
            NPV = ((t1.model1.max_NPV_mut-t1.model1.min_NPV_mut)*t1.model1.flow_p_npv+t1.model1.min_NPV_mut)#/(t1.model1.max_NPV_mut-t1.model1.min_NPV_mut)
            NPV2  = (sum((t1.Harvested_V_log.loc[s,r,year]+t1.Harvested_V_pulp.loc[s,r,year]-t1.DITCHMAINT.loc[s,r,year])/((1+rr)**(2.5+year-2016)) * t1.model1.X1[(s,r)] for (s,r) in t1.model1.index1 for year in t1.years)+sum(t1.PV.loc[s,r,max(t1.years)]/((1+rr)**(max(t1.years)-2016)) * t1.model1.X1[(s,r)] for (s,r) in t1.model1.index1))#/(t1.model1.max_NPV_mut-t1.model1.min_NPV_mut)
            return NPV2 == NPV
        
        t1.model1.flow_p_PEAT = Param(default=1, mutable=True)
        
        def PEAT_bound_lower_rule(model1):
            PEAT = ((t1.model1.max_PEAT_mut-t1.model1.min_PEAT_mut)*t1.model1.flow_p_PEAT+t1.model1.min_PEAT_mut)/(t1.model1.max_PEAT_mut-t1.model1.min_PEAT_mut)
            PEAT2  = (-t1.model1.PEAT_CO2_EQV+t1.model1.BM_CO2_EQV)/(t1.model1.max_PEAT_mut-t1.model1.min_PEAT_mut)
            #UPDATED NEXT TWO
            PEAT = ((t1.model1.max_PEAT_mut-t1.model1.min_PEAT_mut)*t1.model1.flow_p_PEAT+t1.model1.min_PEAT_mut)#/(t1.model1.max_PEAT_mut-t1.model1.min_PEAT_mut)
            PEAT2  = (t1.model1.PEAT_CO2_EQV)#/(t1.model1.max_PEAT_mut-t1.model1.min_PEAT_mut)
            
            PEAT = -1*PEAT #/100
            PEAT2 = -1*PEAT2 #/100
            return PEAT2 == PEAT
        
        
        try:
            t1.model1.del_component(t1.model1.OBJ)   
        except:
            print("OK")
        
        #DISCOUNT RATE
        rr = 0.03
        
        if reg_type == "ALL":
            t1.model1.max_INC_mut = range_inc_npv.loc['NONE','max_INC']
            t1.model1.min_NPV_mut = range_inc_npv.loc['NONE','min_NPV']
            t1.model1.max_NPV_mut = range_inc_npv.loc['NONE','max_NPV']
            t1.model1.min_PEAT_mut = range_inc_npv.loc['NONE','min_PEAT']
            t1.model1.max_PEAT_mut = range_inc_npv.loc['NONE','max_PEAT']
        elif reg_type == "FC":
            t1.model1.max_INC_mut = range_inc_npv.loc['FC','max_INC']
            t1.model1.min_NPV_mut = range_inc_npv.loc['FC','min_NPV']
            t1.model1.max_NPV_mut = range_inc_npv.loc['FC','max_NPV']
            t1.model1.min_PEAT_mut = range_inc_npv.loc['FC','min_PEAT']
            t1.model1.max_PEAT_mut = range_inc_npv.loc['FC','max_PEAT']
        elif reg_type == "SC":
            t1.model1.max_INC_mut = range_inc_npv.loc['SC','max_INC']
            t1.model1.min_NPV_mut = range_inc_npv.loc['SC','min_NPV']
            t1.model1.max_NPV_mut = range_inc_npv.loc['SC','max_NPV']
            t1.model1.min_PEAT_mut = range_inc_npv.loc['SC','min_PEAT']
            t1.model1.max_PEAT_mut = range_inc_npv.loc['SC','max_PEAT']
        
        try:
            t1.model1.del_component(t1.model1.NPV_lower)    
        except:
            "Print NO NPV_Lower"
        try:
            t1.model1.del_component(t1.model1.PEAT_lower)    
        except:
            "Print NO PEAT_Lower"  
        try:
            t1.model1.del_component(t1.model1.INC_lower) 
        except:
            "Print NO INC_Lower"
        
        if OPT == "INC_PEAT":
            def outcome_rule(model1):
                NPV = sum((t1.Harvested_V_log.loc[s,r,year]+t1.Harvested_V_pulp.loc[s,r,year]-t1.DITCHMAINT.loc[s,r,year])/((1+rr)**(2.5+year-2016)) * t1.model1.X1[(s,r)] for (s,r) in t1.model1.index1 for year in t1.years)+sum(t1.PV.loc[s,r,max(t1.years)]/((1+rr)**(max(t1.years)-2016)) * t1.model1.X1[(s,r)] for (s,r) in t1.model1.index1)
                PEAT = -t1.model1.PEAT_CO2_EQV#+t1.model1.BM_CO2_EQV
                return (t1.model1.DEC_inc)*(1-t1.model1.flow_p_OBJ)- t1.model1.flow_p_OBJ*((PEAT-t1.model1.min_PEAT_mut)/(t1.model1.max_PEAT_mut- t1.model1.min_PEAT_mut))
            t1.model1.OBJ = Objective(rule=outcome_rule, sense=maximize)
            t1.model1.NPV_lower = Constraint(rule=NPV_bound_lower_rule)
            t1.model1.INC_bounded = Constraint(t1.years,rule=INC_bounded_rule)
            
        elif OPT == "NPV_INC":
            def outcome_rule(model1):
                NPV = sum((t1.Harvested_V_log.loc[s,r,year]+t1.Harvested_V_pulp.loc[s,r,year]-t1.DITCHMAINT.loc[s,r,year])/((1+rr)**(2.5+year-2016)) * t1.model1.X1[(s,r)] for (s,r) in t1.model1.index1 for year in t1.years)+sum(t1.PV.loc[s,r,max(t1.years)]/((1+rr)**(max(t1.years)-2016)) * t1.model1.X1[(s,r)] for (s,r) in t1.model1.index1)
                PEAT = -t1.model1.PEAT_CO2_EQV#+t1.model1.BM_CO2_EQV
                return ((NPV-t1.model1.min_NPV_mut)/(t1.model1.max_NPV_mut- t1.model1.min_NPV_mut))*(1-t1.model1.flow_p_OBJ)+t1.model1.flow_p_OBJ*(t1.model1.DEC_inc)
            t1.model1.OBJ = Objective(rule=outcome_rule, sense=maximize)
            t1.model1.PEAT_lower = Constraint(rule=PEAT_bound_lower_rule)
            t1.model1.INC_bounded = Constraint(t1.years,rule=INC_bounded_rule)
        
        elif OPT == "NPV_PEAT":
            def outcome_rule(model1):
                NPV = sum((t1.Harvested_V_log.loc[s,r,year]+t1.Harvested_V_pulp.loc[s,r,year]-t1.DITCHMAINT.loc[s,r,year])/((1+rr)**(2.5+year-2016)) * t1.model1.X1[(s,r)] for (s,r) in t1.model1.index1 for year in t1.years)+sum(t1.PV.loc[s,r,max(t1.years)]/((1+rr)**(max(t1.years)-2016)) * t1.model1.X1[(s,r)] for (s,r) in t1.model1.index1)
                PEAT = -t1.model1.PEAT_CO2_EQV#+t1.model1.BM_CO2_EQV
                return ((NPV-t1.model1.min_NPV_mut)/(t1.model1.max_NPV_mut- t1.model1.min_NPV_mut))*(1-t1.model1.flow_p_OBJ)-t1.model1.flow_p_OBJ*((PEAT-t1.model1.min_PEAT_mut)/(t1.model1.max_PEAT_mut- t1.model1.min_PEAT_mut))#+0.00000001*t1.model1.DEC_inc
            t1.model1.OBJ = Objective(rule=outcome_rule, sense=maximize)
            t1.model1.INC_lower = Constraint(t3.years,rule=INC_bound_lower_rule)
        
        data_only = {}
        data_only_harv={}
        data_only_harv_peat={}
        land_use={}
        land_use_peat={}
        EMISSIONS={}
        EMISSIONS_RR={}
        EMISSIONS_RR
        DEVEL_CLASS_EM = {}
        PEAT_EMISSIONS_PERIODIC ={}
        decs ={}
        
        try:
            t1.model1.del_component(t1.model1.regime_limit_FC)
        except:
            print("OK*")
        try:
            t1.model1.del_component(t1.model1.regime_limit_SC)
        except:
            print("OK*")
        
        def regime_rule_SC(model1, t):
            row_sum = sum(model1.X1[(t,p)] for p in [x[1] for x in model1.index1SC if x[0] == t])
            return row_sum == 1
        def regime_rule_FC(model1, t):
            row_sum = sum(model1.X1[(t,p)] for p in [x[1] for x in model1.index1FC if x[0] == t])
            return row_sum == 1
        
        if reg_type == "SC":
            t1.model1.regime_limit_SC = Constraint(t1.model1.stands, rule=regime_rule_SC)
        elif reg_type == "FC":
            t1.model1.regime_limit_FC = Constraint(t1.model1.stands, rule=regime_rule_FC)
        else:
            print("NO restriction")
        counter = 0
        t1.model1.flow_p_npv = 0
        t1.model1.flow_p_inc = 0
        t1.model1.flow_p_PEAT = 0
        for flow_PEAT in range(0,21):
            for flow_OBJ in range(0,21):
                counter = counter+1
                print(counter)
                DATA1 = DATA
                if OPT == "NPV_INC":
                    t1.model1.flow_p_PEAT = flow_PEAT*0.05
                elif  OPT == "NPV_PEAT":
                    t1.model1.flow_p_inc = flow_PEAT*0.05
                elif OPT == "INC_PEAT":
                    t1.model1.flow_p_npv = flow_PEAT*0.05
                t1.model1.flow_p_OBJ = flow_OBJ*0.05
                t1.solve()
                if (t1.results.solver.status == SolverStatus.ok) and (t1.results.solver.termination_condition == TerminationCondition.optimal):
                    b1 = pd.DataFrame(t1.model1.X1)
                    b1["Dec"] = 0.0
                    for l in range(0,len(b1)):
                        b1["Dec"][l] = t1.model1.X1[(b1[0][l],b1[1][l])].value
                    b1 = b1.rename(columns = {0:"id",1:"branching_group","Dec":"DEC"})
                    b1.set_index(["id","branching_group"],inplace=True)
                    result = DATA1.join(b1, how='inner')
                    result_P1 = result.loc[(result.index.get_level_values('year') == 2016)]
                    result_P2 = result.loc[(result.index.get_level_values('year') == 2021)]
                    result_P10 = result.loc[(result.index.get_level_values('year') == max(t1.years))]
                    NPV_X = sum((result['Harvested_V_log'] + result['Harvested_V_pulp']-result['DITCHMAINT'])/((1+rr)**(2.5+result['Y1']-2016)) * result['DEC'])+sum(result_P10['PV']/((1+rr)**(max(t1.years)-2016)) * result_P10['DEC'])
                    PV_X = sum(result_P10['PV']/((1+rr)**(max(t1.years)-2016)) * result_P10['DEC'])
                    INC_X = min([sum((result.loc[(slice(None),slice(None),year),"Harvested_V_log"]+ result.loc[(slice(None),slice(None),year),'Harvested_V_pulp'])*result.loc[(slice(None),slice(None),year),'DEC']) for year in t1.years])
                    CO2 = sum(result['CO2_MEAN']* result['DEC'])
                    N2O = sum(result['N2O_MEAN']* result['DEC'])
                    CH4 = sum(result['CH4_MEAN']* result['DEC'])
                    
                    
                    NPV_X_ha = sum((result['Harvested_V_log'] + result['Harvested_V_pulp']-result['DITCHMAINT'])/((1+rr)**(2.5+result['Y1']-2016)) * result['DEC'])+sum(result_P10['PV']/((1+rr)**(max(t1.years)-2016)) * result_P10['DEC'])/sum(DATA1["AREA_ALL"])
                    PV_X_ha = sum(result_P10['PV']/((1+rr)**(max(t1.years)-2016)) * result_P10['DEC'])/sum(DATA1["AREA_ALL"])
                    INC_X_ha = min([sum((result.loc[(slice(None),slice(None),year),"Harvested_V_log"]+ result.loc[(slice(None),slice(None),year),'Harvested_V_pulp'])*result.loc[(slice(None),slice(None),year),'DEC']) for year in t1.years])/sum(result.loc[(slice(None),slice(None),2016.0),'AREA_ALL']*result.loc[(slice(None),slice(None),2016.0),'DEC'])
                    HARV_X = sum(result['H_V'] * result['DEC'])
                    HARV_X_ha = sum(result['H_V'] * result['DEC'])/sum(DATA1["AREA_ALL"])
                    HARV_X_log = sum(result['H_V_LOG'] * result['DEC'])
                    HARV_X_ha_log = sum(result['H_V_LOG'] * result['DEC'])/sum(DATA1["AREA_ALL"])
                    HARV_X_pulp = sum(result['H_V_PULP'] * result['DEC'])
                    HARV_X_ha_pulp = sum(result['H_V_PULP'] * result['DEC'])/sum(DATA1["AREA_ALL"])
                    data_only[flow_PEAT*1000+flow_OBJ]=[NPV_X,INC_X,-t1.model1.PEAT_CO2_EQV.value+t1.model1.BM_CO2_EQV.value,t1.model1.C_soil_CO2_EQV.value,t1.model1.BM_CO2_EQV.value,t1.model1.PEAT_CO2_EQV_100.value,NPV_X_ha,INC_X_ha,-t1.model1.PEAT_CO2_EQV.value+t1.model1.BM_CO2_EQV.value,t1.model1.DEC_inc.value*t1.model1.max_INC_mut.value,t1.model1.PEAT_CO2_EQV.value,t1.model1.BM_CO2_EQV.value,flow_PEAT,CO2,N2O,CH4,HARV_X,HARV_X_ha,HARV_X_log,HARV_X_ha_log,HARV_X_pulp,HARV_X_ha_pulp]
                    Harv = [sum((result.loc[(result.index.get_level_values('year') == year)]['Harvested_V_log']+result.loc[(result.index.get_level_values('year') == year)]['Harvested_V_pulp']) * result.loc[(result.index.get_level_values('year') == year)]['DEC']) for year in t1.years]
                    Harv_PEAT = [sum((result.loc[(result.index.get_level_values('year') == year)]['Harvested_V_log']+result.loc[(result.index.get_level_values('year') == year)]['Harvested_V_pulp']) * result.loc[(result.index.get_level_values('year') == year)]['DEC']*result.loc[(result.index.get_level_values('year') == year)]['PEAT']) for year in t1.years]
                    data_only_harv[flow_PEAT*1000+flow_OBJ] = Harv
                    data_only_harv_peat[flow_PEAT*1000+flow_OBJ] = Harv_PEAT
                    decs[flow_PEAT*1000+flow_OBJ] = b1 
        return data_only, data_only_harv, data_only_harv_peat, decs

    import matplotlib
    import numpy as np

    data_only_ALL_NPV, data_only_harv_ALL_NPV, data_only_harv_peat_ALL_NPV, data_only_ALL_decs = limit_regimes(t3,"ALL",trade_off)
    data_all_pd_NPV = pd.DataFrame(data_only_ALL_NPV).transpose()
    data_all_pd_NPV.columns = ['ALL_NPV','ALL_INC','ALL_PEAT_CO2_EKV','ALL_C_Soil','ALL_BM_total','ALL_PEAT_CO2_EKV_RAND','NPV_X_ha','INC_X_ha','PEAT',"INC_DEC","PEAT_CO2","BM_CO2","const_flow","CO2","N2O","CH4","HARV_X","HARV_X_ha","HARV_X_LOG","HARV_X_ha_LOG","HARV_X_PULP","HARV_X_ha_PULP"] 

    data_all_pd_NPV.to_csv(path_output + "ALLNPV_DATA_2_"+constraint+"x"+RG+".csv")

    for k in data_only_ALL_decs.keys():
        data_only_ALL_decs[k]["ITER"]= k
    pd.concat([data_only_ALL_decs[v] for v in list(data_only_ALL_decs.keys())]).to_csv(path_output+"ALLNPV_DATA_2_"+constraint+"x"+RG+"_DECS.csv")
