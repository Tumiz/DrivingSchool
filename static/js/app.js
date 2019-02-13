new Vue({
    delimiters: ['${', '}'],
    el: '#app',
    data: function () {
        return {
            pickedObj: {
                id: 0,
                wheel_base: 2.7,
                v: 0,
                front_wheel_angle: 0,
                x: 0,
                y: 0,
                a: 0,
                ai:  false,
                greedy:10,
                success_counts: 0,
                failure_counts: 0,
            },
            need_response:false,
            time:0,
            timer: 0,
            number: 0,
            ws: []
        }
    },
    mounted() {
        var canvas = document.getElementById("canvas")
        canvas.onwheel = this.wheelHandler
        canvas.onmousedown = this.mouseDownHandler
        canvas.onmouseup = this.mouseUpHandler
        canvas.onmousemove = this.mouseMoveHandler
        canvas.ondblclick = this.mouseDoubleClickHandler
        document.onkeydown = this.keyDownHandler
        document.onkeyup = this.keyUpHandler
        this.connect()
    },
    methods: {
        formatTooltip(val) {
            return val / 100
        },
        send(type, data) {
            var request = {
                type: type,
                data: data
            }
            this.ws.send(JSON.stringify(request))
        },
        onMessage(event) {
            var msg = JSON.parse(event.data)
            var type = msg.type
            var data = msg.data
            switch (type) {
                case "timer":
                    if(this.need_response){
                        var obj=objects.getObjectById(data.id)
                        if(obj&&obj.type=="Car"){
                            obj.a=data.a
                            obj.step(100)
                        }
                        this.need_response=false
                    }
                    break
                default:
                    break
            }
        },
        connect(func) {
            var ws = new WebSocket(window.location.href.replace("http", "ws") + "sim")
            ws.onmessage = this.onMessage
            ws.onopen = function () {
                var request = {
                    type: "status",
                    data: "client ready"
                }
                ws.send(JSON.stringify(request))
            }
            this.ws = ws
        },
        start() {
            if (this.ws.readyState == 3)
                this.connect()
            if (this.ws.readyState == 1 && this.timer == 0)
                this.timer = setInterval(this.onTimer, 100)
            this.need_response=false
        },
        onTimer() {
            if(!this.need_response){
                for (var i = 0, l = objects.children.length; i < l; i++) {
                    var obj = objects.children[i]
                    if (obj.type == "Car") {
                        if(obj.ai){
                            R=0
                            var x_gap=obj.position.x-flag.position.x
                            if(Math.abs(x_gap)>50){
                                this.pickedObj.failure_counts+=1
                                R=-1
                                obj.reset()
                            }
                            else if(Math.abs(x_gap)<1&&Math.abs(obj.v)<0.1){
                                this.pickedObj.success_counts+=1
                                R=1
                            }
                            data={
                                id:obj.id,
                                x_gap:x_gap,
                                v:obj.v,
                                R:R,
                                t:this.time,
                                greedy:this.pickedObj.greedy/100
                            }
                            this.send("timer",data)
                            this.need_response=true
                            this.time+=1
                        }     
                    }
                }
            }
            if(pickedObj){
                this.pickedObj.x=pickedObj.position.x.toPrecision(4)
                this.pickedObj.y=pickedObj.position.y.toPrecision(4)
                if (pickedObj.type == "Car"){
                    this.pickedObj.v = pickedObj.v
                    this.pickedObj.a = pickedObj.a
                    this.pickedObj.front_wheel_angle=pickedObj.front_wheel_angle
                }
            }
        },
        reset() {
            scene.add(cameras[0])
            while (objects.children.length > 0){
                var obj=objects.children[0]
                if(obj.type=="Car")
                    objects.remove(obj)
            }
            this.number = objects.children.length
            this.send("reset", "")
        },
        stop() {
            clearInterval(this.timer)
            this.timer = 0
        },
        open(url) {
            window.open(url)
        },
        changeAIstate(enable){
            if(pickedObj && pickedObj.type == "Car")
                pickedObj.ai=enable
        },
        wheelHandler(event) {
            activeViewPort = currentViewPort(event.offsetX, event.offsetY)
            camera = cameras[activeViewPort]
            switch (activeViewPort) {
                case 0:
                    lookAtCenter.translateOnAxis(zAxis, event.deltaY / 12)
                    camera.translateOnAxis(zAxis, -event.deltaY / 12)
                    break
                case 1:
                    camera.zoom = Math.max(1, camera.zoom + event.deltaY / 12)
                    camera.updateProjectionMatrix()
                    break
                default:
                    break
            }
        },
        mouseDownHandler(event) {
            activeViewPort = currentViewPort(event.offsetX, event.offsetY)
            pickedObj = pickObj(mouse, camera)
            preX = event.offsetX
            preY = event.offsetY
            if (pickedObj == null) {
                if (event.button == 0 && event.ctrlKey == 1) {
                    point = pickPoint(mouse, camera)
                    if (point !== null) {
                        pickedObj = new Car()
                        pickedObj.position.copy(point)
                        pickedObj.add(cameras[0])
                        objects.add(pickedObj)
                        this.number = objects.children.length
                    }
                }
            }else{
                this.pickedObj.id=pickedObj.id
                this.pickedObj.x=pickedObj.position.x.toPrecision(4)
                this.pickedObj.y=pickedObj.position.y.toPrecision(4)
                if(pickedObj.type=="Car"){
                    this.pickedObj.ai=pickedObj.ai
                }
            }
        },
        mouseUpHandler(event) {

        },
        mouseMoveHandler(event) {
            if (event.buttons != 1) {
                return
            }
            activeViewPort = currentViewPort(event.offsetX, event.offsetY)
            var dx = event.offsetX - preX
            var dy = event.offsetY - preY
            preX = event.offsetX
            preY = event.offsetY
            switch (activeViewPort) {
                case 0:
                    if (event.shiftKey == 1) {
                        var distance = lookAtCenter.position.length()
                        camera.translateOnAxis(zAxis, -distance)
                        if (Math.abs(dy) > Math.abs(dx)) {
                            camera.rotateOnAxis(xAxis, -dy / 1000)
                        } else {
                            camera.rotateOnWorldAxis(zAxis, -dx / 1000)
                        }
                        camera.translateOnAxis(zAxis, distance)
                    } else {
                        camera.translateOnAxis(xAxis, -dx / 10)
                        camera.translateOnAxis(yAxis, dy / 10)
                    }
                    return
                case 1:
                    if (pickedObj) {
                        point = pickPoint(mouse, camera)
                        pickedObj.position.x = point.x
                        pickedObj.position.y = point.y
                        this.pickedObj.x = point.x.toPrecision(4)
                        this.pickedObj.y = point.y.toPrecision(4)
                    }
                    else {
                        camera.translateOnAxis(xAxis, -dx / camera.zoom)
                        camera.translateOnAxis(yAxis, dy / camera.zoom)
                    }
                    return
                default:
                    return
            }
        },
        mouseDoubleClickHandler(event) {
            activeViewPort = currentViewPort(event.offsetX, event.offsetY)
            switch (activeViewPort) {
                case 0:
                    camera.lookAt(0, 0, 0)
                    lookAtCenter.position.set(0, 0, -camera.position.length())
                    break
                case 1:
                    camera.position.set(0, 0, 60)
                    camera.zoom = 1
                    camera.lookAt(0, 0, 0)
                    break
                default:
                    break
            }
        },
        keyDownHandler(event) {
            switch (event.keyCode) {
                case 81://Q
                    pickedObj.rotation.z += 0.01
                    break
                case 69://E
                    pickedObj.rotation.z -= 0.01
                    break
                case 46://delete
                case 8://back space
                    if(pickedObj){
                        scene.add(cameras[0])
                        objects.remove(pickedObj)
                        this.pickedObj.id=0
                        this.number=objects.children.length
                    }
                    break
                case 87://w
                    if (pickedObj && pickedObj.type == "Car") {
                        if (this.timer == 0) {
                            this.start()
                        }
                        if (pickedObj.a < 0)
                            pickedObj.a = 0
                        else
                            pickedObj.a += 0.01
                        this.pickedObj.a = pickedObj.a
                    }
                    break
                case 83://S:
                    if (pickedObj && pickedObj.type == "Car") {
                        if (this.timer == 0) {
                            this.start()
                        }
                        if (pickedObj.a > 0)
                            pickedObj.a = 0
                        else
                            pickedObj.a -= 0.01
                        this.pickedObj.a = pickedObj.a
                    }
                    break
                case 68://D:
                    if (pickedObj && pickedObj.type == "Car") {
                        if (this.timer == 0) {
                            this.start()
                        }
                        if (pickedObj.front_wheel_angle > 0)
                            pickedObj.front_wheel_angle = 0
                        else
                            pickedObj.front_wheel_angle -= 0.01
                        this.pickedObj.front_wheel_angle = pickedObj.front_wheel_angle
                    }
                    break
                case 65://A:
                    if (pickedObj && pickedObj.type == "Car") {
                        if (this.timer == 0) {
                            this.start()
                        }
                        if (pickedObj.front_wheel_angle < 0)
                            pickedObj.front_wheel_angle = 0
                        else
                            pickedObj.front_wheel_angle += 0.01
                        this.pickedObj.front_wheel_angle = pickedObj.front_wheel_angle
                    }
                    break
                default:
                    break
            }
        },
        keyUpHandler(event) {
        }
    }
})