#coding=utf-8 
import tornado.web
import tornado.httpserver
import tornado.options
import os
import json
import time
import visdom
from car import Car
from ai import AI
from tornado.ioloop import PeriodicCallback,IOLoop
from tornado.web import RequestHandler,StaticFileHandler
from tornado.websocket import WebSocketHandler
from tornado.options import define, options

class Session():
    def __init__(self):
        self.cars=dict()
        self.timer=PeriodicCallback(self.on_timer,50)
        self.users=set()
        self.viz=visdom.Visdom()
        self.plot_velocity=[]
        self.plot_round=[]

    def publish(self,type,data):
        msg={"type":type,"data":data}
        for u in self.users:
            u.write_message(msg)

    def on_timer(self):
        for id in self.cars:
            car=self.cars[id]
            car.step()
            if(self.plot_velocity):
                self.viz.line(X=car.t_history,Y=car.v_history,win=self.plot_velocity)
            else:
                self.plot_velocity=self.viz.line(X=car.t_history,Y=car.v_history)
            if abs(car.v)<0.001 and abs(car.x-30)<1:
                car.success_counts+=1
                if(not self.plot_round):
                    self.plot_round=self.viz.line(X=[car.success_counts],Y=[car.t])
                else:
                    self.viz.line(X=[car.success_counts],Y=[car.t],win=self.plot_round,update="append")
                car.reset()
            elif car.x<-10 or car.x>50:
                car.failure_counts+=1
                car.reset()
            elif car.ai.enable:
                car.v=car.ai.decision(car.x-30,car.v)
            self.publish("timer",car.dict())

    def reset(self):
        self.stop()
        self.cars.clear()

    def start(self):
        if not self.timer.is_running():
            self.timer.start()
            if self.timer.is_running():
                print("session started")
            else:
                print("session failed to started")
        self.publish("cars number",len(self.cars))

    def stop(self):
        self.timer.stop()
        if(not self.timer.is_running()):
            print("session stopped")
        else:
            print("session failed to stop")

class IndexHandler(RequestHandler):
    def get(self):
        if(self.request.uri=="/"):
            self.render("index.html")
            print(self.request)

class SimHandler(WebSocketHandler):
    
    session=Session()
    def open(self):
        self.session.users.add(self)
        print("open websocket",len(self.session.users),"users")

    def on_message(self, message):
        msg=json.loads(message)
        request_type=msg["type"]
        if(request_type=="start"): 
            id=int(msg["data"]["id"])
            if(self.session.cars.__contains__(id)):
                car=self.session.cars[id]
                car.wheelbase=float(msg["data"]["wheel_base"])
                car.v=float(msg["data"]["velocity"])
                car.front_wheel_angle=float(msg["data"]["front_wheel_angle"])
                car.a=float(msg["data"]["a"])
                car.x=float(msg["data"]["x"])
                car.y=float(msg["data"]["y"])
                car.ai.enable=bool(msg["data"]["ai"]["enable"])               
            self.session.start()
        elif(request_type=="status"):
            for id in self.session.cars:
                car=self.session.cars[id]
                self.session.publish("status",car.dict())      
        elif(request_type=="create"):
            id=int(msg["data"]["id"])
            if(self.session.cars.__contains__(id)):
                car=self.session.cars[id]
                car.x=float(msg["data"]["x"])
                car.y=float(msg["data"]["y"])
            else:
                self.session.cars[id]=Car()
                self.session.cars[id].id=id
            self.session.publish("create",self.session.cars[id].dict())
        elif(request_type=="control"):
            id=int(msg["data"]["id"])
            car=self.session.cars[id]
            car.front_wheel_angle=float(msg["data"]["front_wheel_angle"])
            car.a=float(msg["data"]["a"])  
            self.session.start()   
        elif(request_type=="reset"):
            self.session.reset()
        elif(request_type=="stop"):
            self.session.stop()

    def on_close(self):
        self.session.users.remove(self) # 用户关闭连接后从容器中移除用户

    def check_origin(self, origin):
        return True  # 允许WebSocket的跨域请求

if __name__ == '__main__':
    tornado.options.parse_command_line()
    app = tornado.web.Application([
            (r"/", IndexHandler),
            (r"/sim", SimHandler),
            (r"/(.*)",StaticFileHandler)      
        ],
        static_path = os.path.join(os.path.dirname(__file__), "static"),
        template_path = os.path.join(os.path.dirname(__file__), "template"),
        debug = True
        )
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(8080)
    IOLoop.current().start()
