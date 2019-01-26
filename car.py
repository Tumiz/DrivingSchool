import time
from math import sin,cos,tan

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
        self.t=-1

    def step(self):
        current_time=time.time()
        if(self.t==-1):
            dt=0
        else:
            dt=current_time-self.t
        self.vx=self.v*cos(self.yaw)
        self.vy=self.v*sin(self.yaw)
        self.x+=self.vx*dt
        self.y+=self.vy*dt
        new_v=self.v+self.a*dt
        if(new_v*self.v<0):
            self.v=0
            self.a=0
        else:
            self.v=new_v
        self.yaw+=self.v/self.wheelbase*tan(self.front_wheel_angle)*dt
        self.t=current_time