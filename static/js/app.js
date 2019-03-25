new Vue({
    delimiters: ['${', '}'],
    el: '#app',
    data: function () {
        return {
            id: 0,
            wheel_base: 2.7,
            v: 0,
            w: 0,
            x: 0,
            y: 0,
            a: 0,
            ai: true,
            rand_target: false,
            p_error: 6,//position error
            counts: 0,
            running: false,
            time: 0,
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
        flag = new Flag(this.p_error)
        flag.position.set(20, 0, 0)
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
            if (!this.running)
                return
            var msg = JSON.parse(event.data)
            var type = msg.type
            var data = msg.data
            switch (type) {
                case "timer":
                    var obj = objects.getObjectById(data.id)
                    if (obj && obj.type == "Car") {
                        obj.a = data.a
                        obj.w = data.w
                        obj.step(100)
                        this.x = obj.position.x.toFixed(2)
                        this.y = obj.position.y.toFixed(2)
                        this.v = obj.v
                        this.a = obj.a
                        this.w = obj.w
                        this.time += 1
                        var done = this.sendstate(obj)
                        if (done) {
                            obj.reset()
                            this.time = 0
                            this.counts += 1
                            if (this.rand_target) {
                                var r = 20
                                var theta = Math.random() * 1.2 - 0.6
                                flag.position.x = r * Math.cos(theta)
                                flag.position.y = r * Math.sin(theta)
                            }
                        }
                    }
                    break
                default:
                    break
            }
        },
        sendstate(obj) {
            var done=false
            if (this.time > 150) {
                done= true
            }
            var local = obj.worldToLocal(flag.position.clone())
            this.send("timer", {
                done: done,
                id: obj.id,
                x: local.x,
                y: local.y,
                rz: obj.rotation.z,
                v: obj.v,
                p_error: this.p_error
            })
            return done
        },
        connect(func) {
            var ws = new WebSocket(window.location.href.replace("http", "ws") + "sim")
            ws.onmessage = this.onMessage
            ws.onopen = function () {
                var request = {
                    type: "reset",
                    data: "client ready"
                }
                ws.send(JSON.stringify(request))
            }
            this.ws = ws
        },
        start() {
            this.running = true
            if (this.ws.readyState == 3)
                this.connect()
            for (var i = 0, l = objects.children.length; i < l; i++) {
                var obj = objects.children[i]
                if (obj.type == "Car") {
                    this.sendstate(obj)
                }
            }
        },
        stop() {
            this.running = false
        },
        reset() {
            this.running = false
            for (var i = 0, l = objects.children.length; i < l; i++) {
                var obj = objects.children[i]
                if (obj.type == "Car") {
                    obj.reset()
                }
            }
            this.number = objects.children.length
            this.time = 0
            this.counts = 0
            this.send("reset", {})
        },
        open(url) {
            window.open(url)
        },
        changeperror(value) {
            flag.circle.geometry = new THREE.CircleGeometry(value, 32)
        },
        changeAIstate(enable) {
            if (pickedObj && pickedObj.type == "Car")
                pickedObj.ai = enable
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
                        this.number = objects.children.length
                    }
                }
            } else {
                this.id = pickedObj.id
                this.x = pickedObj.position.x.toPrecision(4)
                this.y = pickedObj.position.y.toPrecision(4)
                if (pickedObj.type == "Car") {
                    this.ai = pickedObj.ai
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
                        this.x = point.x.toPrecision(4)
                        this.y = point.y.toPrecision(4)
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
                    if (pickedObj) {
                        scene.add(cameras[0])
                        objects.remove(pickedObj)
                        this.id = 0
                        this.number = objects.children.length
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
                        this.a = pickedObj.a
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
                        this.a = pickedObj.a
                    }
                    break
                case 68://D:
                    if (pickedObj && pickedObj.type == "Car") {
                        if (this.timer == 0) {
                            this.start()
                        }
                        if (pickedObj.w > 0)
                            pickedObj.w = 0
                        else
                            pickedObj.w -= 0.01
                        this.w = pickedObj.w
                    }
                    break
                case 65://A:
                    if (pickedObj && pickedObj.type == "Car") {
                        if (this.timer == 0) {
                            this.start()
                        }
                        if (pickedObj.w < 0)
                            pickedObj.w = 0
                        else
                            pickedObj.w += 0.01
                        this.w = pickedObj.w
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