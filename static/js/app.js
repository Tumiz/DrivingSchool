new Vue({
    delimiters: ['${', '}'],
    el: '#app',
    data: function () {
        return {
            car: {
                id:0,
                wheel_base: 2.7,
                velocity: 1,
                front_wheel_angle: 0,
                x:0,
                y:0,
                a:0,
                ai:{
                    enable:true,
                },
                success_counts:0,
                failure_counts:0,
            },
            number:0,
            ws: [],
            chart: {
                velocity: [],
                a: []
            }
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
            switch(type){
                case "timer":
                    var obj = objects.getObjectById(data.id)
                    if (!obj && car) {
                        obj = car.clone()
                        objects.add(obj)
                    }
                    if (obj) {
                        obj.position.x = data.x
                        obj.position.y = data.y
                        obj.rotation.z = data.yaw
                    }
                    if (data.id == this.car.id) {
                        this.car.velocity = data.v
                        this.car.a = data.a
                        this.car.front_wheel_angle = data.front_wheel_angle
                        this.car.x = data.x.toPrecision(4)
                        this.car.y = data.y.toPrecision(4)
                        this.car.success_counts=data.success_counts
                        this.car.failure_counts=data.failure_counts
                        this.number= objects.children.length
                    }
                    break
                case "create":
                    if (data.id == this.car.id) {
                        this.car.velocity = data.v
                        this.car.a = data.a
                        this.car.front_wheel_angle = data.front_wheel_angle
                    }
                    break
                case "objects number":
                    this.number=data
                    break
                case "status":
                    var obj = objects.getObjectById(data.id)
                    if (!obj && car) {
                        obj = car.clone()
                        objects.add(obj)
                    }
                    if (obj) {
                        obj.position.x = data.x
                        obj.position.y = data.y
                        obj.rotation.z = data.yaw
                    }
                    break
                default:
                    break
            }
        },
        connect(func) {

            var ws = new WebSocket(window.location.href.replace("http","ws")+"sim")
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
            if (this.ws.readyState == 1)
                this.send("start", this.car)
        },
        reset() {
            while (objects.children.length > 0)
            objects.remove(objects.children[0])
            this.number = objects.children.length
            this.send("reset", "")
        },
        stop() {
            this.send("stop", "")
        },
        open(url){
            window.open(url)
        },
        wheelHandler(event) {
            activeViewPort = currentViewPort(event.offsetX, event.offsetY)
            console.log(activeViewPort, event.deltaY)
            camera = cameras[activeViewPort]
            switch (activeViewPort) {
                case 0:
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
            var obj = pickObj(mouse, camera)
            preX = event.offsetX
            preY = event.offsetY
            if (obj == null && event.button == 0 && event.ctrlKey == 1) {
                point = pickPoint(mouse, camera)
                console.log("pick", JSON.stringify(point))
                if (point !== null) {
                    obj = car.clone()
                    obj.position.copy(point)
                    objects.add(obj)
                    this.number = objects.children.length
                    this.send("create", {
                        id: obj.id,
                        x: obj.position.x,
                        y: obj.position.y,
                    })
                }
            }
            if (obj !== null) {
                onPickedObj = true
                pickedObj = obj
                this.car.id = obj.id
                this.send("create", {
                    id: obj.id,
                    x: obj.position.x,
                    y: obj.position.y,
                })
            }
        },
        mouseUpHandler(event) {
            onPickedObj = false
        },
        mouseMoveHandler(event) {
            if(event.buttons!=1){
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
                        var vector=camera.position.clone()
                        var distance = camera.position.clone().sub(lookAtCenter.position).length()
                        camera.translateOnAxis(zAxis, -distance)
                        if (Math.abs(dy) > Math.abs(dx)) {
                            console.log("y")
                            camera.rotateOnAxis(xAxis,-dy / 1000)
                        } else {
                            console.log("x")
                            camera.rotateOnAxis(yAxis,-dx / 1000)
                        }
                        camera.translateOnAxis(zAxis, distance)
                    }else{
                        lookAtCenter.position.sub(camera.position)
                        camera.translateOnAxis(xAxis, -dx / 10)
                        camera.translateOnAxis(yAxis, dy / 10)
                        lookAtCenter.position.add(camera.position)
                    }
                    return
                case 1:
                    if (onPickedObj) {
                        point = pickPoint(mouse, camera)
                        pickedObj.position.x=point.x
                        pickedObj.position.y=point.y
                        this.car.x = point.x.toPrecision(4)
                        this.car.y = point.y.toPrecision(4)
                        this.send("create", {
                            id: this.car.id,
                            x: this.car.x,
                            y: this.car.y,
                        })
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
            console.log("dbclick")
            activeViewPort = currentViewPort(event.offsetX, event.offsetY)
            if (activeViewPort==1&&event.button === 0) {
                camera.position.set(0, 0, 60)
                camera.lookAt(0,0,0)
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
                    console.log("remove", pickedObj.id)
                    markers.remove(pickedObj)
                    break
                case 87://w
                    if(this.car.a<0)
                        this.car.a=0
                    else
                        this.car.a+=0.01
                    this.send("control", this.car)
                    break
                case 83://S:
                    if(this.car.a>0)
                        this.car.a=0
                    else
                        this.car.a-=0.01
                    this.send("control", this.car)
                    break
                case 68://D:
                    if(this.car.front_wheel_angle>0)
                        this.car.front_wheel_angle=0
                    else
                        this.car.front_wheel_angle-=0.01
                    this.send("control", this.car)
                    break
                case 65://A:
                    if(this.car.front_wheel_angle<0)
                        this.car.front_wheel_angle=0
                    else
                        this.car.front_wheel_angle+=0.01
                    this.send("control", this.car)
                    break
                default:
                    break
            }
        },
        keyUpHandler(event) {   
        }
    }
})