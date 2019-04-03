#coding=utf-8 
import tornado.web
import tornado.httpserver
import tornado.options
import os
import json
import time

from tornado.ioloop import IOLoop
from tornado.web import RequestHandler,StaticFileHandler
from tornado.websocket import WebSocketHandler
from tornado.options import define, options

from agents.continuouspolicygradient import Agent

class IndexHandler(RequestHandler):
    def get(self):
        if(self.request.uri=="/"):
            self.render("index.html")
            print(self.request)

class SimHandler(WebSocketHandler):
    
    agents=dict()
    users=set()

    def publish(self,type,data):
        msg={"type":type,"data":data}
        for u in self.users:
            u.write_message(msg)

    def open(self):
        self.users.add(self)
        print("open websocket",len(self.users),"users")

    def on_message(self, message):
        msg=json.loads(message)
        request_type=msg["type"]
        request_data=msg["data"]
        if(request_type=="timer"): 
            done=bool(request_data["done"])
            id=int(request_data["id"])# id of agent
            v=float(request_data["v"])
            x=float(request_data["x"])
            y=float(request_data["y"])
            rz=float(request_data["rz"])
            p_error=float(request_data["p_error"])
            if(not self.agents.__contains__(id)):
                self.agents[id]=Agent()
            response_data=dict()
            response_data["id"]=id
            response_data["a"],response_data["w"]=self.agents[id].decision(done,x,y,rz,v,p_error)
            self.publish("timer",response_data)
        elif(request_type=="reset"):
            self.agents.clear()
            print("agents number:",len(self.agents))
                
    def on_close(self):
        self.users.remove(self) # 用户关闭连接后从容器中移除用户

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
    print("Driving School is ready. Now you can navigate to http://localhost:8080")
    IOLoop.current().start()
