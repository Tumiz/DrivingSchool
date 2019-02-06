import time
from math import sin,cos,tan,isnan
from ai import AI,Controller,Planner
class Car():

    def __init__(self):
        self.id=0
        self.front_wheel_angle=0
        self.wheelbase=2.7
        self.yaw =0
        self.a=0
        self.x=0
        self.y=0
        self.v=0
        self.vx=0
        self.vy=0
        self.t=0
        self.t_start=-1
        self.ai=Planner()
        self.success_counts = 0
        self.failure_counts = 0
        self.v_history=[]
        self.t_history=[]

    def dict(self):
        c=self.__dict__.copy()
        del c["ai"]
        return c

    def step(self):
        if(self.t_start==-1):
            dt=0
            self.t_start=time.time()
        else:
            dt=time.time()-self.t-self.t_start
        print(dt,self.v)
        self.vx=self.v*cos(self.yaw)
        self.vy=self.v*sin(self.yaw)
        self.x+=self.vx*dt
        self.y+=self.vy*dt
        self.v+=self.a*dt
        self.yaw+=self.v/self.wheelbase*tan(self.front_wheel_angle)*dt
        self.t+=dt
        self.v_history.append(self.v)
        if(len(self.v_history)>300):
            del self.v_history[0]
        self.t_history.append(self.t)
        if(len(self.t_history)>300):
            del self.t_history[0]

    def reset(self):
        self.x=0
        self.v=0
        self.a=0
        self.t=0
        self.t_start=-1
        self.ai.x_gap=0
        self.ai.v_target=0
        self.v_history=[]
        self.t_history=[]