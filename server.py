#coding=utf-8 
import tornado.web
import tornado.httpserver
import tornado.options
import os
import json
import time
from ai import AI
from tornado.ioloop import IOLoop
from tornado.web import RequestHandler,StaticFileHandler
from tornado.websocket import WebSocketHandler
from tornado.options import define, options

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
            id=int(request_data["id"])
            v=float(request_data["v"])
            t=float(request_data["t"])
            greedy=float(request_data["greedy"])
            x_gap=float(request_data["x_gap"])
            R=float(request_data["R"])
            if(self.agents.__contains__(id)):
                response_data=dict()
                response_data["id"]=id
                response_data["a"]=self.agents[id].decision(x_gap,v,R,t,greedy)
                self.publish("timer",response_data)
            else:
                self.agents[id]=AI()
        elif(request_type=="reset"):
            self.agents.clear()

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
