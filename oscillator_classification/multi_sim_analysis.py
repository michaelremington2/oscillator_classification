#!/usr/bin/python
import pandas as pd
import numpy as np
import csv
import os
import time
import single_sim_analysis as ipc


class Experiment_Per_Cycle_Analysis(object):
    def __init__(self,exp_name, per_cycle_data_file,parameter_file,output_file_csv,nash_inf_count_sensitivity=5,nash_inf_mag_sensitivity=0.05):
        self.per_cycle = pd.read_csv(per_cycle_data_file,header = 0, index_col=None)
        self.parameters = pd.read_csv(parameter_file,header = 0, index_col=None)
        self.output_file_csv = output_file_csv
        self.exp_name = exp_name
        self.nash_inf_count_sensitivity=nash_inf_count_sensitivity
        self.nash_inf_mag_sensitivity=nash_inf_mag_sensitivity
        self.per_gen_data_format()
        header = ['exp_name',
                  'sim_id',
                  'exp',
                  'org',
                  'count_str_flip', 
                  'mean_strategy_flip_time', 'std_strategy_flip_time', 'var_strategy_flip_time',
                  'mean_magnitude_of_strategy_flip', 'std_magnitude_of_strategy_flip', 'var_magnitude_of_strategy_flip','nash_val']
        if(os.path.exists(self.output_file_csv) and os.path.isfile(self.output_file_csv)):
            pass
        else:
            self.create_csv(fp=self.output_file_csv,header=header)

    def create_csv(self,fp, header=None):
        with open(fp, "w") as my_empty_csv:
            pass
        if header is not None:
           self.append_data(fp = fp, d_row = header)

    def append_data(self,fp,d_row):
        with open(fp, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(d_row)

    def by_gen_filter(self,krat_reproduction_freq, snake_reproduction_freq):
        by_gen_filter_col = []
        for index,row in self.per_cycle.iterrows():
            if row['org']=='krat' and (row['cycle'] % krat_reproduction_freq==0):
                val=1
            elif row['org']=='snake' and (row['cycle'] % snake_reproduction_freq==0):
                val=1
            else:
                val=0
            by_gen_filter_col.append(val)
        return by_gen_filter_col

    def per_gen_data_format(self):
        self.krat_reproduction_freq = self.parameters['krat_reproduction_freq_per_x_cycles'].max()
        self.snake_reproduction_freq = self.parameters['snake_reproduction_freq_per_x_cycles'].max()
        self.per_cycle['by_gen_filter'] = self.by_gen_filter(krat_reproduction_freq = self.krat_reproduction_freq,
                                                             snake_reproduction_freq = self.snake_reproduction_freq)

        self.by_gen_data = self.per_cycle[self.per_cycle['by_gen_filter']==1]

    def nash(self,pc_count, pc_mag_value, sensitivity_points, sensitivity_magnitude):
        if pc_count <= sensitivity_points and pc_mag_value <= sensitivity_magnitude:
            nash=1
        else:
            nash=0
        return nash

    def run_classifacation_algorithm(self,sim):
        step_size=1
        sensitivity=0.01
        x = list(sim['generation'])
        y = list(sim["bush_pw_mean"])
        yx_dx = ipc.aprox_derv_central(data=y,step_size_forward=1,step_size_backward=step_size)
        smooth_yx_dx = ipc.savitzky_golay(y_list=yx_dx, window_size=3, order=1, deriv=0, rate=1)
        yx_dx2 = ipc.aprox_derv_central(data=yx_dx,step_size_forward=step_size,step_size_backward=step_size)
        smooth_yx_dx2 = ipc.savitzky_golay(y_list=yx_dx2, window_size=3, order=1, deriv=0, rate=1)
        data_tuples = list(zip(x,y,smooth_yx_dx,smooth_yx_dx2))
        df = pd.DataFrame(data_tuples, columns=['x','y','yx_dx','yx_dx2'])
        classifacation = ipc.inflection_point_classifaction(df=df,sensitivity=sensitivity)
        df['point_classifacation'] = classifacation
        dot_class = ipc.plotable_points(df=df)
        df['dot_class'] = dot_class
        df['min_max_flip'] = ipc.min_max_flip_by_gen(df=df)
        #df['time_as_max_min'] = df.groupby((df['min_max_flip'] != df['min_max_flip'].shift(1)).cumsum()).cumcount()+1
        df['time_as_max_min'] = ipc.time_as_min_max(df=df)
        df['min_max_filter'] = ipc.min_max_filter(df=df)
        mean_strategy_flip_time = df[df['min_max_filter']==1]['time_as_max_min'].mean()
        std_strategy_flip_time = df[df['min_max_filter']==1]['time_as_max_min'].std()
        var_strategy_flip_time = df[df['min_max_filter']==1]['time_as_max_min'].var()
        number_of_inf_points = df['min_max_filter'].sum()
        if len(df[df['point_classifacation'].isin(['min','max'])]['y'])==0:
            mean_magnitude_of_strategy_flip = 0
            std_magnitude_of_strategy_flip = 0
            var_magnitude_of_strategy_flip = 0
        else:
            magnitude_data = abs(np.diff(df[df['point_classifacation'].isin(['min','max'])]['y'],1))
            mean_magnitude_of_strategy_flip = magnitude_data.mean()
            std_magnitude_of_strategy_flip = magnitude_data.std()
            var_magnitude_of_strategy_flip = magnitude_data.var()
        nash_val = self.nash(pc_count=number_of_inf_points,
                         pc_mag_value=mean_magnitude_of_strategy_flip, 
                         sensitivity_points=self.nash_inf_count_sensitivity, 
                         sensitivity_magnitude=self.nash_inf_mag_sensitivity)
        sim_id = sim['sim_id'].max()
        org = sim['org'].max()
        exp = sim['experiment'].max()
        return [self.exp_name,sim_id,exp,org,number_of_inf_points, mean_strategy_flip_time, std_strategy_flip_time, var_strategy_flip_time, mean_magnitude_of_strategy_flip,std_magnitude_of_strategy_flip,var_magnitude_of_strategy_flip,nash_val]


    def loop_through_sims(self):
        sim_groupings = self.by_gen_data.groupby(['sim_id','org'])
        for name, sim in sim_groupings:
            data = self.run_classifacation_algorithm(sim=sim)
            self.append_data(fp = self.output_file_csv,d_row=data)



if __name__=="__main__":
    main()
