var scene,renderer,grid,cars,lines,camera,xAxis,yAxis,zAxis,point,points,raycaster,lookAtCenter,preX,preY,container,car
var mousePressed=false
var onPickedObj=false
var cameras=new Array
var pickedObj=null
var mouse = new THREE.Vector2()
var origin=new THREE.Vector3(0,0,0)
var activeViewPort=0
var grid
var colors
var mode=0
var vertices=new Array
var fortime=0,arraytime=0
init();
animate();

function init() {
    colors={
        "Pedestrian":new THREE.Color("skyblue"),
        "Motobike":new THREE.Color("white"),
        "StaticObstacle":new THREE.Color("grey"),
        "Car":new THREE.Color("#00ff00"),
        "undefined":new THREE.Color("#00ff00"),
        "selected":new THREE.Color("red")
    }

    xAxis=new THREE.Vector3(1,0,0)
    yAxis=new THREE.Vector3(0,1,0)
    zAxis=new THREE.Vector3(0,0,1)

    raycaster = new THREE.Raycaster()
    raycaster.near=5
    raycaster.far=100
    raycaster.params.Points.threshold=1

    renderer = new THREE.WebGLRenderer( { antialias: true } );
    container = document.getElementById("canvas")
    renderer.setSize( container.offsetWidth, container.offsetHeight);
    renderer.autoClear=false
    container.appendChild(renderer.domElement)

    scene = new THREE.Scene()
    LoadCar()
    loadMap()
    cameras[0] = new THREE.PerspectiveCamera( 70, container.offsetWidth / container.offsetHeight, 0.01, 6000 )
    cameras[0].position.set(-4,0,2)
    cameras[0].up.set(0,0,1)
    cameras[0].lookAt(1000,0,0)
    cameras[0].zoom=1
    lookAtCenter=origin.clone()

    cameras[1] = new THREE.OrthographicCamera( container.offsetWidth/-4, container.offsetWidth/4, container.offsetHeight/2, container.offsetHeight/-2, 0.01, 6000 )
    cameras[1].position.set(0,0,60)
    cameras[1].lookAt(origin)
    cameras[1].zoom=5
    cameras[1].updateProjectionMatrix()

    var light = new THREE.AmbientLight( 0xffffff )
    scene.add(light)

    cars=new THREE.Object3D()
    grid=new Grid(1000,200)
    plane=new Plane(1000)

    points=new THREE.Points()
    points.material=new THREE.PointsMaterial({color:0x409EFF,sizeAttenuation:false,size:2})

    scene.add(cars,new Axis(),grid,points)

    activeViewPort=1
    camera=cameras[1]

    window.addEventListener( 'resize', onContainerResize, false );
}

function animate() {
    requestAnimationFrame( animate )
    render()
}

function onContainerResize() {
    var aspect=container.offsetWidth / container.offsetHeight
    cameras[0].aspect = aspect
    cameras[0].updateProjectionMatrix()
    cameras[1].aspect = aspect*0.5
    cameras[1].updateProjectionMatrix()
    renderer.setPixelRatio(aspect);
    renderer.setSize(container.offsetWidth, container.offsetHeight);
}

function render() {
    renderer.clear()
    renderer.setViewport( 0, 0, container.offsetWidth/2, container.offsetHeight/2 )
    renderer.render(scene, cameras[0])
    renderer.setViewport( container.offsetWidth/2, 0, container.offsetWidth/2, container.offsetHeight )
    renderer.render(scene, cameras[1])
}

function pickObj(mouse,camera){
    raycaster.setFromCamera( mouse, camera )
    var intersects = raycaster.intersectObjects(cars.children,true)
    console.log(intersects.length)
    if(intersects.length>0){
        var obj=intersects[0].object.parent
        obj.add(cameras[0])
        return obj
    }
    else
        return null
}

function pickPoint(mouse,camera){
    raycaster.setFromCamera( mouse, camera )
    var intersects = raycaster.intersectObject(plane)
    if(intersects.length>0)
        return intersects[0].point
    return null
}
function setClass(type){
    if(pickedObj!==null){
        pickedObj.material.color=colors[type]
        pickedObj.class=type
    }
}

function currentViewPort(x,y){
    if(x<container.offsetWidth/2)
    {
        if(y<container.offsetHeight/2)
        {
            mouse.x = 4*x / container.offsetWidth - 1;
            mouse.y = - 4*y / container.offsetHeight + 1
            camera=cameras[0]
            return 0
        }
        else
        {
            return 2
        }
    }
    else
    {
        mouse.x = ( 4*x-2*container.offsetWidth)/ container.offsetWidth - 1;
        mouse.y = - 2*y / container.offsetHeight + 1
        camera=cameras[1]
        return 1
    }
}
function directionFromMouse(mouse,camera){
    return new THREE.Vector3(mouse.x,mouse.y,0.5).unproject(camera)
}

function Plane(size){
    var geometry = new THREE.PlaneGeometry( size, size );
    geometry.computeBoundingBox()
    var material = new THREE.MeshBasicMaterial( {side: THREE.DoubleSide,transparent:true,opacity:0} );
    var plane = new THREE.Mesh( geometry, material );
    return plane
}
function Grid(size,divs){
    var obj= new THREE.GridHelper( size, divs, "#828282", "#333333" );
    obj.rotation.x=Math.PI/2
    return obj
}
function Axis(){
    var g_cylinder_z = new THREE.CylinderGeometry( 0, 0.3, 1);
    var material = new THREE.MeshLambertMaterial( {color: "blue"} );
    var cylinder = new THREE.Mesh( g_cylinder_z, material );
    cylinder.position.set(0,0,4)
    cylinder.rotation.x=Math.PI/2
    var line_z=new LineSegment(
                [0, 0, 0],
                [0, 0, 4],
                {color: "blue",linewidth:2}
                );
    var line_y = new LineSegment(
                [0, -2, 0],
                [0, 4, 0],
                {color: "#00FF7F",linewidth:2}
                )
    var line_x=new LineSegment(
                [-2, 0, 0],
                [4, 0, 0],
                {color: "red",linewidth:2}
                )
    return new THREE.Object3D().add(cylinder,line_x,line_y,line_z)
}
function Line(vertices,color){
    var geometry=new THREE.Geometry()
    geometry.vertices=vertices
    var material=new THREE.LineBasicMaterial({color:color,linewidth:2})
    var obj=new THREE.Line(geometry,material)
    obj.type="line"
    console.log("new Line",obj.id)
    return obj
}

function LineSegment(p1,p2,materialJson){
    var g=new THREE.Geometry()
    var m=new THREE.LineBasicMaterial(materialJson)
    g.vertices.push(new THREE.Vector3(p1[0],p1[1],p1[2]),new THREE.Vector3(p2[0],p2[1],p2[2]))
    return new THREE.Line(g,m)
}

function LoadCar(position){
    var loader = new THREE.ObjectLoader();
    loader.load(
        "static/res/car.json",
        function ( obj ) {
            car=obj
        },
        function ( xhr ) {
            console.log( (xhr.loaded / xhr.total * 100) + '% loaded' );
        },
        function ( err ) {
            console.error( 'An error happened' );
        }
    );
}

function loadMap(){
    vertices=new Array
    vertices.push(
        new THREE.Vector3(0,-2,0),
        new THREE.Vector3(20,-2,0),
        new THREE.Vector3(30,3,0))
    line1=Line(vertices,"#eeeeee")
    scene.add(line1)
}
