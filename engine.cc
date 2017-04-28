#include "easy_image.hh"
#include "ini_configuration.hh"
#include "l_parser.hh"
#include "vector.hh"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <list>
#include <cmath>
#include <stack>
#include <limits>
#include <map>
#include <set>
#include <assert.h>

class ColorDouble;
class Point2D;
class Line2D;

class ColorDouble{
public:
    ColorDouble(){}
    ColorDouble(double r, double g, double b){
        red = r;
        green = g;
        blue = b;
    }
    double red;
    double green;
    double blue;
};

int roundToInt(double d){
    if(d < 0){
        return (int) std::ceil(d - 0.5);
    }else{
        return (int) std::floor(d + 0.5);
    }
}

//======================================================================================================//
//===============================================2D-ClASSES=============================================//

class Point2D{
public:
    double x;
    double y;
    //z stores the original z-value in case of z-buffering
    double z;
    Point2D(){}
};

class Line2D{
public:
    Point2D p1;
    Point2D p2;
    ColorDouble color;
    double z1;
    double z2;
};

typedef std::list<Line2D> Lines2D;
typedef std::vector<Point2D> Points2D;

//======================================================================================================//
//===============================================3D-CLASSES=============================================//

class Face{
public:
    std::vector<int> point_indexes;
};

class Figure{
public:
    std::vector<Vector3D> points;
    std::vector<Face> faces;
    ColorDouble ambientReflection;
    ColorDouble diffuseReflection;
    ColorDouble specularReflection;
    double reflectionCoefficient;
};

class Light{
public:
    ColorDouble ambientLight;
    ColorDouble diffuseLight;
    ColorDouble specularLight;
    bool infinite = true;
    bool specular = false;
    Vector3D vector;
};

class ZBuffer{
    std::vector<std::vector<double>> zBuffer;
public:
    ZBuffer(const unsigned int width, const unsigned int height){
        double posInf = std::numeric_limits<double>::infinity();
        for(int h = 0; h < height; h++){
            zBuffer.push_back({});
            for(int w = 0; w < width; w++){
                zBuffer[h].push_back(posInf);
            }
        }
    }
    double& operator()(unsigned int x, unsigned int y){
        //return reference to place in vector, so the value can be modified
        return zBuffer[y].at(x);
    }
};

typedef std::vector<Light> Lights3D;

typedef std::vector<Figure> Figures3D;

//======================================================================================================//
//===============================================3D-Figures=============================================//

Figure createCube(){
    Figure cube = Figure();
    std::vector<double> x = {1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0};
    std::vector<double> y = {-1.0,1.0,1.0,-1.0,1.0,-1.0,-1.0,1.0};
    std::vector<double> z = {-1.0,-1.0,1.0,1.0,-1.0,-1.0,1.0,1.0};
    std::vector<int> p1 = {0,4,1,5,6,0};
    std::vector<int> p2 = {4,1,5,0,2,5};
    std::vector<int> p3 = {2,7,3,6,7,1};
    std::vector<int> p4 = {6,2,7,3,3,4};
    for(int i = 0; i < 8; i++){
        Vector3D newPoint = Vector3D();
        newPoint.x = x[i];
        newPoint.y = y[i];
        newPoint.z = z[i];
        cube.points.push_back(newPoint);
    }
    for(int i = 0; i < 6; i++){
        Face newFace = Face();
        newFace.point_indexes = {p1[i], p2[i], p3[i], p4[i]};
        cube.faces.push_back(newFace);
    }
    return cube;
}

Figure createTetrahedron(){
    Figure tetrahedron = Figure();
    std::vector<double> x = {1.0,-1.0,1.0,-1.0};
    std::vector<double> y = {-1.0,1.0,1.0,-1.0};
    std::vector<double> z = {-1.0,-1.0,1.0,1.0};
    std::vector<int> p1 = {0,1,0,0};
    std::vector<int> p2 = {1,3,3,2};
    std::vector<int> p3 = {2,2,1,3};
    for(int i = 0; i < 4; i++){
        Vector3D newPoint = Vector3D();
        newPoint.x = x[i];
        newPoint.y = y[i];
        newPoint.z = z[i];
        tetrahedron.points.push_back(newPoint);
    }
    for(int i = 0; i < 4; i++){
        Face newFace = Face();
        newFace.point_indexes = {p1[i], p2[i], p3[i]};
        tetrahedron.faces.push_back(newFace);
    }
    return tetrahedron;
}

Figure createOctahedron(){
    Figure octahedron = Figure();
    std::vector<double> x = {1.0,0.0,-1.0,0.0,0.0,0.0};
    std::vector<double> y = {0.0,1.0,0.0,-1.0,0.0,0.0};
    std::vector<double> z = {0.0,0.0,0.0,0.0,-1.0,1.0};
    std::vector<int> p1 = {0,1,2,3,1,2,3,0};
    std::vector<int> p2 = {1,2,3,0,0,1,2,3};
    std::vector<int> p3 = {5,5,5,5,4,4,4,4};
    for(int i = 0; i < 6; i++){
        Vector3D newPoint = Vector3D();
        newPoint.x = x[i];
        newPoint.y = y[i];
        newPoint.z = z[i];
        octahedron.points.push_back(newPoint);
    }
    for(int i = 0; i < 8; i++){
        Face newFace = Face();
        newFace.point_indexes = {p1[i], p2[i], p3[i]};
        octahedron.faces.push_back(newFace);
    }
    return octahedron;
}

Figure createIcosahedron(){
    Figure icosahedron = Figure();
    std::vector<int> p1 = {0,0,0,0,0,1,2,2,3,3,4,4,5,5,1,11,11,11,11,11};
    std::vector<int> p2 = {1,2,3,4,5,6,6,7,7,8,8,9,9,10,10,7,8,9,10,6};
    std::vector<int> p3 = {2,3,4,5,1,2,7,3,8,4,9,5,10,1,6,6,7,8,9,10};
    for(int i = 0; i < 20; i++){
        Face newFace = Face();
        newFace.point_indexes = {p1[i], p2[i], p3[i]};
        icosahedron.faces.push_back(newFace);
    }

    Vector3D point1 = Vector3D();
    point1.x = 0.0;
    point1.y = 0.0;
    point1.z = sqrt(5.0)/2;
    icosahedron.points.push_back(point1);
    for(int i = 0; i < 5; i++){
        Vector3D newPoint = Vector3D();
        newPoint.x = cos(i*2*M_PI/5);
        newPoint.y = sin(i*2*M_PI/5);
        newPoint.z = 0.5;
        icosahedron.points.push_back(newPoint);
    }
    for(int i = 0; i < 5; i++){
        Vector3D newPoint = Vector3D();
        newPoint.x = cos(M_PI/5 + i*2*M_PI/5);
        newPoint.y = sin(M_PI/5 + i*2*M_PI/5);
        newPoint.z = -0.5;
        icosahedron.points.push_back(newPoint);
    }
    Vector3D point12 = Vector3D();
    point12.x = 0.0;
    point12.y = 0.0;
    point12.z = -sqrt(5.0)/2;
    icosahedron.points.push_back(point12);
    return icosahedron;
}

Vector3D createMiddlePoint(Vector3D &pA, Vector3D &pB){
    Vector3D pM = Vector3D();
    pM.x = (pA.x + pB.x)/2;
    pM.y = (pA.y + pB.y)/2;
    pM.z = (pA.z + pB.z)/2;
    return pM;
}

Figure createSphere(const double radius, const int n){
    Figure icosahedron = createIcosahedron();
    int nrPoints = icosahedron.points.size();
    for(int i = 0; i < n; i++){
        //save initial nr of faces because new faces will be added
        unsigned long nrFaces = icosahedron.faces.size();
        for(int j = 0; j < nrFaces; j++){
            Face &f = icosahedron.faces.at(j);
            std::vector<int> point_indexes = f.point_indexes;
            Vector3D pA = icosahedron.points[point_indexes[0]];
            Vector3D pB = icosahedron.points[point_indexes[1]];
            Vector3D pC = icosahedron.points[point_indexes[2]];
            Vector3D pD = createMiddlePoint(pA, pB);
            Vector3D pE = createMiddlePoint(pA, pC);
            Vector3D pF = createMiddlePoint(pC, pB);
            icosahedron.points.push_back(pD);
            icosahedron.points.push_back(pE);
            icosahedron.points.push_back(pF);
            Face face1 = Face();
            Face face2 = Face();
            Face face3 = Face();
            face1.point_indexes.push_back(point_indexes[0]);
            face1.point_indexes.push_back(nrPoints);
            face1.point_indexes.push_back(nrPoints + 1);
            face2.point_indexes.push_back(point_indexes[1]);
            face2.point_indexes.push_back(nrPoints + 2);
            face2.point_indexes.push_back(nrPoints);
            face3.point_indexes.push_back(point_indexes[2]);
            face3.point_indexes.push_back(nrPoints + 1);
            face3.point_indexes.push_back(nrPoints + 2);
            //The current face is reused
            f.point_indexes[0] = nrPoints;
            f.point_indexes[1] = nrPoints + 2;
            f.point_indexes[2] = nrPoints + 1;
            icosahedron.faces.push_back(face1);
            icosahedron.faces.push_back(face2);
            icosahedron.faces.push_back(face3);
            //nrPoints has to be adjusted now
            nrPoints += 3;
        }
    }
    for(Vector3D &v: icosahedron.points){
        double r = sqrt(pow(v.x, 2) + pow(v.y, 2) + pow(v.z, 2));
        v.x /= r;
        v.y /= r;
        v.z /= r;
    }
    return icosahedron;
}

Figure createDodecadron(){
    Figure icosahedron = createIcosahedron();
    std::vector<Vector3D> points = icosahedron.points;
    Figure dodecahedron = Figure();
    for(int i = 0; i < 20; i++){
        Vector3D newPoint = Vector3D();
        Face currentFace = icosahedron.faces[i];
        Vector3D p1 = points[currentFace.point_indexes[0]];
        Vector3D p2 = points[currentFace.point_indexes[1]];
        Vector3D p3 = points[currentFace.point_indexes[2]];
        //calculate the centroid of the current face
        newPoint.x = (p1.x + p2.x + p3.x)/3;
        newPoint.y = (p1.y + p2.y + p3.y)/3;
        newPoint.z = (p1.z + p2.z + p3.z)/3;
        dodecahedron.points.push_back(newPoint);
    }
    //icosahedron can now be deleted
    std::vector<int> p1 = {0,0,1,2,3,4,19,19,18,17,16,15};
    std::vector<int> p2 = {1,5,7,9,11,13,18,14,12,10,8,6};
    std::vector<int> p3 = {2,6,8,10,12,14,17,13,11,9,7,5};
    std::vector<int> p4 = {3,7,9,11,13,5,16,12,10,8,6,14};
    std::vector<int> p5 = {4,1,2,3,4,0,15,18,17,16,15,19};
    for(int i = 0; i < 12; i++){
        Face newFace = Face();
        newFace.point_indexes = {p1[i], p2[i], p3[i], p4[i], p5[i]};
        dodecahedron.faces.push_back(newFace);
    }
    return dodecahedron;
}

Figure createCone(const int n, const double h){
    Figure cone = Figure();
    //bottom points
    for(int i = 0; i < n; i++){
        Vector3D newPoint = Vector3D();
        newPoint.x = cos(2*i*M_PI/n);
        newPoint.y = sin(2*i*M_PI/n);
        cone.points.push_back(newPoint);
    }
    //top point
    Vector3D pn = Vector3D();
    pn.z = h;
    cone.points.push_back(pn);
    for(int i = 0; i < n; i++){
        Face newFace = Face();
        newFace.point_indexes = {i, (i+1)%n, n};
        cone.faces.push_back(newFace);
    }
    Face nFace = Face();
    for(int i = 0; i < n; i++){
        nFace.point_indexes.push_back(i);
    }
    cone.faces.push_back(nFace);
    return cone;
}

Figure createCylinder(const int n, const double h){
    Figure cylinder = Figure();
    //bottom points
    for(int i = 0; i < n; i++){
        Vector3D newPoint = Vector3D();
        newPoint.x = cos(2*i*M_PI/n);
        newPoint.y = sin(2*i*M_PI/n);
        cylinder.points.push_back(newPoint);
    }
    //top points
    for(int i = 0; i < n; i++){
        Vector3D newPoint = Vector3D();
        newPoint.x = cos(2*i*M_PI/n);
        newPoint.y = sin(2*i*M_PI/n);
        newPoint.z = h;
        cylinder.points.push_back(newPoint);
    }
    for(int i = 0; i < n; i++){
        Face newFace = Face();
        newFace.point_indexes = {i, (i+1)%n, n+(i+1)%n, n+i};
        cylinder.faces.push_back(newFace);
    }
    for(int i = 0; i < 2; i++){
        Face newFace = Face();
        for(int j = 0; j < n; j++){
            newFace.point_indexes.push_back(j + n*i);
        }
        cylinder.faces.push_back(newFace);
    }
    return cylinder;
}

Figure createTorus(const double r, const double R, const int n, const int m){
    Figure torus = Figure();
    //calculate u and v to save time
    double u = 2*M_PI/n;
    double v = 2*M_PI/m;
    for(int i = 0; i < n; i++){
        double currentU = u*i;
        for(int j = 0; j < m; j++){
            Vector3D newPoint = Vector3D();
            double currentV = v*j;
            newPoint.x = (R + r*cos(currentV))*cos(currentU);
            newPoint.y = (R + r*cos(currentV))*sin(currentU);
            newPoint.z = r*sin(currentV);
            torus.points.push_back(newPoint);
        }
    }
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            Face newFace = Face();
            int iIndex = i*m;
            int iMod = ((i+1)%n)*m;
            int jMod = (j+1)%m;
            newFace.point_indexes = {iIndex+j, iMod+j, iMod+jMod, iIndex+jMod};
            torus.faces.push_back(newFace);
        }
    }
    return torus;
}

void calculatePoints(const Vector3D &tV0, const Vector3D &tV1, Vector3D &v0, Vector3D &v1){
    double oneThird_X = std::abs(tV0.x - tV1.x) / 3;
    double oneThird_Y = std::abs(tV0.y - tV1.y) / 3;
    double oneThird_Z = std::abs(tV0.z - tV1.z) / 3;
    v0.x = tV0.x < tV1.x ? tV0.x + oneThird_X : tV0.x - oneThird_X;
    v1.x = tV0.x < tV1.x ? tV1.x - oneThird_X : tV1.x + oneThird_X;
    v0.y = tV0.y < tV1.y ? tV0.y + oneThird_Y : tV0.y - oneThird_Y;
    v1.y = tV0.y < tV1.y ? tV1.y - oneThird_Y : tV1.y + oneThird_Y;
    v0.z = tV0.z < tV1.z ? tV0.z + oneThird_Z : tV0.z - oneThird_Z;
    v1.z = tV0.z < tV1.z ? tV1.z - oneThird_Z : tV1.z + oneThird_Z;
}

Figure createBuckyball(){
    Figure icosahedron = createIcosahedron();
    Figure buckyball = Figure();
    std::map<std::set<int>, std::vector<int>> hexagonPoints;
    int currentPoint = 0;
    for(Face &f: icosahedron.faces){
        Face hexagon = Face();
        Vector3D v0, v1, v2, v3, v4, v5;
        int tV0_index = f.point_indexes[0];
        int tV1_index = f.point_indexes[1];
        int tV2_index = f.point_indexes[2];
        Vector3D tV0 = icosahedron.points[tV0_index];
        Vector3D tV1 = icosahedron.points[tV1_index];
        Vector3D tV2 = icosahedron.points[tV2_index];
        std::set<int> searchKey = {tV0_index};
        searchKey.insert(tV1_index);
        if(hexagonPoints.find(searchKey) == hexagonPoints.end()){
            calculatePoints(tV0, tV1, v0, v1);
            buckyball.points.push_back(v0);
            buckyball.points.push_back(v1);
            hexagonPoints[searchKey].push_back(currentPoint);
            hexagonPoints[searchKey].push_back(currentPoint + 1);
            hexagon.point_indexes.push_back(currentPoint);
            hexagon.point_indexes.push_back(currentPoint + 1);
            currentPoint += 2;
        }else{
            std::vector<int> currentPoints = hexagonPoints[{tV0_index, tV1_index}];
            //the two points have to be added in reverse order to maintain counter clock wise points
            hexagon.point_indexes.push_back(currentPoints[1]);
            hexagon.point_indexes.push_back(currentPoints[0]);
        }
        searchKey = {tV1_index};
        searchKey.insert(tV2_index);
        if(hexagonPoints.find(searchKey) == hexagonPoints.end()){
            calculatePoints(tV1, tV2, v2, v3);
            buckyball.points.push_back(v2);
            buckyball.points.push_back(v3);
            hexagonPoints[searchKey].push_back(currentPoint);
            hexagonPoints[searchKey].push_back(currentPoint + 1);
            hexagon.point_indexes.push_back(currentPoint);
            hexagon.point_indexes.push_back(currentPoint + 1);
            currentPoint += 2;
        }else{
            std::vector<int> currentPoints = hexagonPoints[{tV1_index, tV2_index}];
            //the two points have to be added in reverse order to maintain counter clock wise points
            hexagon.point_indexes.push_back(currentPoints[1]);
            hexagon.point_indexes.push_back(currentPoints[0]);
        }
        searchKey = {tV2_index};
        searchKey.insert(tV0_index);
        if(hexagonPoints.find(searchKey) == hexagonPoints.end()){
            calculatePoints(tV2, tV0, v4, v5);
            buckyball.points.push_back(v4);
            buckyball.points.push_back(v5);
            hexagonPoints[searchKey].push_back(currentPoint);
            hexagonPoints[searchKey].push_back(currentPoint + 1);
            hexagon.point_indexes.push_back(currentPoint);
            hexagon.point_indexes.push_back(currentPoint + 1);
            currentPoint += 2;
        }else{
            std::vector<int> currentPoints = hexagonPoints[{tV2_index, tV0_index}];
            //the two points have to be added in reverse order to maintain counter clock wise points
            hexagon.point_indexes.push_back(currentPoints[1]);
            hexagon.point_indexes.push_back(currentPoints[0]);
        }
        buckyball.faces.push_back(hexagon);
    }
    std::vector<int> p0 = {0, 2, 6, 10,14,18,22,  30,36,42,48,53};
    std::vector<int> p1 = {5, 1, 4, 8, 12,16,21,  28,34,40,46,58};
    std::vector<int> p2 = {9, 19,3, 7, 11,15,49,  26,32,38,44,56};
    std::vector<int> p3 = {13,47,23,29,35,41,52,  25,31,37,43,54};
    std::vector<int> p4 = {17,20,27,33,39,45,24,  51,55,57,59,50};
    for(int i = 0; i < 12; i++){
        Face newFace = Face();
        newFace.point_indexes = {p0[i],p1[i],p2[i],p3[i],p4[i]};
        buckyball.faces.push_back(newFace);
    }
    return buckyball;
}

void triangulate(const Face &f, std::vector<Face> &newFaces){
    std::vector<int> points = f.point_indexes;
    int nrPoints = (int) f.point_indexes.size() - 2;
    Face newFace;
    for(int i = 1; i <= nrPoints; i++){
        newFace = Face();
        newFace.point_indexes = {points[0], points[i], points[i+1]};
        newFaces.push_back(newFace);
    }
}

void triangulateFigure(Figure &figure){
    unsigned long size = figure.faces.size();
    std::vector<Face> newFaces;
    Face currentFace;
    for(Face &currentFace: figure.faces){
        if(currentFace.point_indexes.size() > 3) {
            triangulate(currentFace, newFaces);
        }else{
            newFaces.push_back(currentFace);
        }
    }
    figure.faces = newFaces;
}

Figure create3D_LSystem(const LParser::LSystem3D &l_system, ColorDouble &color){
    //not implemented yet
}

//==========================================================================================================//
//================================================3D-FUNCTIONS==============================================//

Matrix scaleFigure(const double scale){
    Matrix scaleMatrix = Matrix();
    scaleMatrix(1,1) = scale;
    scaleMatrix(2,2) = scale;
    scaleMatrix(3,3) = scale;
    return scaleMatrix;
}

Matrix rotateX(const double angle){
    Matrix rotateMatrix;
    double angleRad = angle * M_PI / 180;
    rotateMatrix(2,2) = std::cos(angleRad);
    rotateMatrix(2,3) = std::sin(angleRad);
    rotateMatrix(3,2) = -std::sin(angleRad);
    rotateMatrix(3,3) = std::cos(angleRad);
    return rotateMatrix;
}

Matrix rotateY(const double angle){
    Matrix rotateMatrix;
    double angleRad = angle * M_PI / 180;
    rotateMatrix(1,1) = std::cos(angleRad);
    rotateMatrix(1,3) = -std::sin(angleRad);
    rotateMatrix(3,1) = std::sin(angleRad);
    rotateMatrix(3,3) = std::cos(angleRad);
    return rotateMatrix;
}

Matrix rotateZ(const double angle){
    Matrix rotateMatrix;
    double angleRad = angle * M_PI / 180;
    rotateMatrix(1,1) = std::cos(angleRad);
    rotateMatrix(1,2) = std::sin(angleRad);
    rotateMatrix(2,1) = -std::sin(angleRad);
    rotateMatrix(2,2) = std::cos(angleRad);
    return rotateMatrix;
}

Matrix translate(const Vector3D &center){
    Matrix translateMatrix;
    translateMatrix(4,1) = center.x;
    translateMatrix(4,2) = center.y;
    translateMatrix(4,3) = center.z;
    return translateMatrix;
}

void applyTransformation(Figure &figure, const Matrix &matrix){
    for(int i = 0; i < figure.points.size(); i++){
        figure.points[i] *= matrix;
    }
}

void toPolar(const Vector3D &point, double &theta, double &phi, double &r){
    r = sqrt(pow(point.x,2) + pow(point.y,2) + pow(point.z,2));
    theta = std::atan2(point.y, point.x);
    phi = std::acos(point.z/r);
}

Matrix eyePointTrans(const Vector3D &eyepoint){
    Matrix eyePTMatrix;
    double theta;
    double phi;
    double r;
    toPolar(eyepoint, theta, phi, r);
    eyePTMatrix(1,1) = -std::sin(theta);
    eyePTMatrix(1,2) = -std::cos(theta) * std::cos(phi);
    eyePTMatrix(1,3) = std::cos(theta) * std::sin(phi);
    eyePTMatrix(2,1) = std::cos(theta);
    eyePTMatrix(2,2) = -std::sin(theta) * std::cos(phi);
    eyePTMatrix(2,3) = std::sin(theta) * std::sin(phi);
    eyePTMatrix(3,2) = std::sin(phi);
    eyePTMatrix(3,3) = std::cos(phi);
    eyePTMatrix(4,3) = -r;
    return eyePTMatrix;
}

void applyTransformation(Figures3D &figures, const Matrix &matrix){
    for(int i = 0; i < figures.size(); i++){
        applyTransformation(figures[i], matrix);
    }
}

Point2D doProjection(const Vector3D &point, const double d){
    Point2D newPoint = Point2D();
    newPoint.x = (d * point.x) / (-point.z);
    newPoint.y = (d * point.y) / (-point.z);
    newPoint.z = point.z;
    return newPoint;
}

Lines2D doProjection(const Figures3D &figures, const double d){
    Lines2D lines;
    for(int i = 0; i < figures.size(); i++){
        Points2D points;
        Figure f = figures[i];
        for(Vector3D &v: f.points){
            points.push_back(doProjection(v, d));
        }
        for(Face &face: f.faces){
            unsigned long size = face.point_indexes.size();
            //if there are only two points, the code in the loop has to be executed only once
            if(size == 2){size = 1;}
            for(int j = 0; j < size; j++){
                Point2D p1;
                Point2D p2;
                // if the face contains more than 2 points, the last point has to be connected to the first
                if(size > 2 and (j + 1) == size){
                    p1 = points[face.point_indexes[0]];
                    p2 = points[face.point_indexes[j]];
                }else{
                    p1 = points[face.point_indexes[j]];
                    p2 = points[face.point_indexes[j+1]];
                }
                Line2D line = Line2D();
                line.p1 = p1;
                line.p2 = p2;
                line.color = f.ambientReflection;
                line.z1 = p1.z;
                line.z2 = p2.z;
                lines.push_back(line);
            }
        }
    }
    return lines;
}

void generateFractal(Figure &fig, Figures3D &fractal, const int nr_iterations, const double scale){
    fractal = {fig};
    Figures3D newFigures;
    Figure newFigure;
    Matrix tM;
    Matrix sM = scaleFigure(1/scale);
    for(int i = 0; i < nr_iterations; i++){
        for(Figure &currentFigure: fractal){
            for(int j = 0; j < currentFigure.points.size(); j++){
                newFigure = currentFigure;
                applyTransformation(newFigure, sM);
                Vector3D translateVector = currentFigure.points[j] - newFigure.points[j];
                tM = translate(translateVector);
                applyTransformation(newFigure, tM);
                newFigures.push_back(newFigure);
            }
        }
        fractal = newFigures;
        newFigures = {};
    }
}

void createSponge(const int nr_iterations, Figures3D &sponge){
    Figure Cube = createCube();
    sponge = {Cube};
    Figures3D newFigures;
    Figure f0, f1, f2, f3, f4;
    Vector3D dummyVector;
    Matrix sM = scaleFigure(1.0/3.0);
    Matrix tM;
    for(int i = 0; i < nr_iterations; i++){
        for(Figure &currentFigure: sponge){
            int faceIter = 0;
            for(int j: {0, 4, 1, 5}){ // 0
                Face currentFace = currentFigure.faces[faceIter];
                // p0 is the current corner, p1 is above it, p2 is to the right and p3 is in the oposite corner
                Vector3D p0 = currentFigure.points[currentFace.point_indexes[0]];
                Vector3D p1 = currentFigure.points[currentFace.point_indexes[3]];
                Vector3D p2 = currentFigure.points[currentFace.point_indexes[1]];
                Vector3D p3 = currentFigure.points[currentFace.point_indexes[2]];
                f0 = currentFigure;
                f1 = currentFigure;
                f2 = currentFigure;
                f3 = currentFigure;
                f4 = currentFigure;
                applyTransformation(f0, sM);
                applyTransformation(f1, sM);
                applyTransformation(f2, sM);
                applyTransformation(f3, sM);
                applyTransformation(f4, sM);
                Vector3D p4, p5, p6, p7, extraPoint1, extraPoint2;
                calculatePoints(p0, p1, p4, extraPoint1);
                calculatePoints(p2, p3, dummyVector, extraPoint2);
                calculatePoints(extraPoint1, extraPoint2, p7, dummyVector);
                calculatePoints(p0, p2, p5, dummyVector);
                tM = translate(p0 - f0.points[j]);
                applyTransformation(f0, tM);
                tM = translate(p4 - f1.points[j]);
                applyTransformation(f1, tM);
                tM = translate(p5 - f2.points[j]);
                applyTransformation(f2, tM);
                tM = translate(extraPoint1 - f3.points[j]);
                applyTransformation(f3, tM);
                tM = translate(p7 - f4.points[j]);
                applyTransformation(f4, tM);
                newFigures.push_back(f0);
                newFigures.push_back(f1);
                newFigures.push_back(f2);
                newFigures.push_back(f3);
                newFigures.push_back(f4);
                faceIter += 1;
            }
        }
        sponge = newFigures;
        newFigures = {};
    }
}

//======================================================================================================//
//===============================================2D-LINES===============================================//

void draw_zbuf_line(ZBuffer &zBuf, img::EasyImage &image, unsigned int x0, unsigned int y0, double z0,
                    unsigned int x1, unsigned int y1, double z1, img::Color color){
    assert(x0 < image.get_width() && y0 < image.get_height());
    assert(x1 < image.get_width() && y1 < image.get_height());

    if(x0 == x1 and y0 == y1){
        //special case if only one pixel has to be drawn
        double zMin = 1 / std::min(z0, z1);
        if(zMin < zBuf(x0, y0)){
            zBuf(x0, y0) = zMin;
            image(x0, y0) = color;
        }
    }else if (x0 == x1){
        //special case for x0 == x1
        int yMin = std::min(y0, y1);
        int yMax = std::max(y0, y1);
        double zMin = yMin == y0 ? z0 : z1;
        double zMax = yMax == y1 ? z1 : z0;
        // a + 1 is the amount of pixels that has to be drawn
        int a = yMax - yMin;
        for (int i = a; i >= 0; i--){
            double p = (double)i/(double)a;
            double zi = (p/zMax) + ((1-p)/zMin);
            if(zi < zBuf(x0, yMin + i)){
                zBuf(x0, yMin + i) = zi;
                image(x0, yMin + i) = color;
            }
        }
    }else if (y0 == y1){
        //special case for y0 == y1
        int xMin = std::min(x0, x1);
        int xMax = std::max(x0, x1);
        double zMin = xMin == x0 ? z0 : z1;
        double zMax = xMax == x1 ? z1 : z0;
        // a + 1 is the amount of pixels that has to be drawn
        int a = xMax - xMin;
        for (int i = a; i >= 0; i--){
            double p = (double)i/(double)a;
            double zi = (p/zMax) + ((1-p)/zMin);
            if(zi < zBuf(xMin + i, y0)){
                zBuf(xMin + i, y0) = zi;
                image(xMin + i, y0) = color;
            }
        }
    }else{
        if (x0 > x1){
            //flip points if x0>x1: we want x0 to have the lowest value
            std::swap(x0, x1);
            std::swap(y0, y1);
            std::swap(z0, z1);
        }
        double m = ((double) y1 - (double) y0) / ((double) x1 - (double) x0);
        if (-1.0 <= m && m <= 1.0){
            // a + 1 is the amount of pixels that has to be drawn
            int a = x1 - x0;
            for (int i = a; i >= 0; i--){
                unsigned int x = x0 + i;
                unsigned int y = (unsigned int) roundToInt(y0 + m * i);
                double p = (double)i/(double)a;
                double zi = (p/z1) + ((1-p)/z0);
                if(zi < zBuf(x, y)){
                    zBuf(x, y) = zi;
                    image(x, y) = color;
                }
            }
        }
        else if (m > 1.0){
            // a + 1 is the amount of pixels that has to be drawn
            int a = y1 - y0;
            for (int i = a; i >= 0; i--){
                unsigned int x = (unsigned int) roundToInt(x0 + (i / m));
                unsigned int y = y0 + i;
                double p = (double)i/(double)a;
                double zi = (p/z1) + ((1-p)/z0);
                if(zi < zBuf(x, y)){
                    zBuf(x, y) = zi;
                    image(x, y) = color;
                }
            }
        }
        else if (m < -1.0){
            // a + 1 is the amount of pixels that has to be drawn
            int a = y0 - y1;
            for (int i = a; i >= 0; i--){
                unsigned int x = (unsigned int) roundToInt(x0 - (i / m));
                unsigned int y = y0 - i;
                double p = (double)i/(double)a;
                double zi = (p/z1) + ((1-p)/z0);
                if(zi < zBuf(x, y)){
                    zBuf(x, y) = zi;
                    image(x, y) = color;
                }
            }
        }
    }
}

void draw_zbuf_triag(ZBuffer &zBuf, img::EasyImage &image, Vector3D &A, Vector3D &B, Vector3D &C,
                     double d, double dx, double dy,
                     ColorDouble ambientReflection,
                     ColorDouble diffuseReflection,
                     ColorDouble specularReflection,
                     double reflectionCoefficient,
                     Lights3D &lights, bool useLight){
    Point2D a2D, b2D, c2D;
    a2D.x = (d*A.x/-(A.z)) + dx;
    a2D.y = (d*A.y/-(A.z)) + dy;
    b2D.x = (d*B.x/-(B.z)) + dx;
    b2D.y = (d*B.y/-(B.z)) + dy;
    c2D.x = (d*C.x/-(C.z)) + dx;
    c2D.y = (d*C.y/-(C.z)) + dy;
    int yMax = roundToInt(std::max(std::max(a2D.y, b2D.y), c2D.y) - 0.5);
    int yMin = roundToInt(std::min(std::min(a2D.y, b2D.y), c2D.y) + 0.5);
    double posInf = std::numeric_limits<double>::infinity();
    double negInf = -std::numeric_limits<double>::infinity();
    double xlAB, xlAC, xlBC, xrAB, xrAC, xrBC;
    int xl, xr;

    //calculate dzdx and dzdy
    double xg = (a2D.x + b2D.x + c2D.x) / 3;
    double yg = (a2D.y + b2D.y + c2D.y) / 3;
    //zg is actually 1/zg
    double zg = 1/(3*A.z) + 1/(3*B.z) + 1/(3*C.z);
    Vector3D u = B - A;
    Vector3D v = C - A;
    Vector3D w = Vector3D::vector(u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x);
    Vector3D n = Vector3D::normalise(w);

    ColorDouble totalColor(0.0, 0.0, 0.0);
    Lights3D pointLights;
    Lights3D specularLights;
    double red, green, blue;
    for(Light &light: lights){
        red = ambientReflection.red * light.ambientLight.red;
        green = ambientReflection.green * light.ambientLight.green;
        blue = ambientReflection.blue * light.ambientLight.blue;
        totalColor.red = (red + totalColor.red) < 1.0 ? red + totalColor.red : 1.0;
        totalColor.green = (green + totalColor.green) < 1.0 ? green + totalColor.green : 1.0;
        totalColor.blue = (blue + totalColor.blue) < 1.0 ? blue + totalColor.blue : 1.0;
        if (light.infinite and useLight){
            Vector3D direction = Vector3D::normalise(-light.vector);
            double cos = (direction.x * n.x) + (direction.y * n.y) + (direction.z * n.z);
            if(cos > 0) {
                red = (diffuseReflection.red * light.diffuseLight.red) * cos;
                green = (diffuseReflection.green * light.diffuseLight.green) * cos;
                blue = (diffuseReflection.blue * light.diffuseLight.blue) * cos;
            }else{
                red = 0.0;
                green = 0.0;
                blue = 0.0;
            }
            totalColor.red = (red + totalColor.red) < 1.0 ? red + totalColor.red : 1.0;
            totalColor.green = (green + totalColor.green) < 1.0 ? green + totalColor.green : 1.0;
            totalColor.blue = (blue + totalColor.blue) < 1.0 ? blue + totalColor.blue : 1.0;
        }else if(!light.infinite and useLight){
            pointLights.push_back(light);
        }
        if(light.specular and useLight){
            specularLights.push_back(light);
        }
    }
    img::Color currentColor = img::Color(totalColor.red*255, totalColor.green*255, totalColor.blue*255);

    double k = w.x*A.x + w.y*A.y + w.z*A.z;
    if(k == 0){
        //the triangle has to be drawn as a line
        unsigned int x0 = roundToInt(std::min(std::min(a2D.x, b2D.x), c2D.x));
        unsigned int y0 = roundToInt(std::min(std::min(a2D.y, b2D.y), c2D.y));
        unsigned int x1 = roundToInt(std::max(std::max(a2D.x, b2D.x), c2D.x));
        unsigned int y1 = roundToInt(std::max(std::max(a2D.y, b2D.y), c2D.y));
        double z0, z1;
        //find the appropriate z0
        if(x0 == a2D.x){z0 = A.z;}
        else if(x0 == b2D.x){z0 = B.z;}
        else{z0 = C.z;}
        //find the appropriate z1
        if(x1 == a2D.x){z1 = A.z;}
        else if(x1 == b2D.x){z1 = B.z;}
        else{z1 = C.z;}
        draw_zbuf_line(zBuf, image, x0, y0, z0, x1, y1, z1, currentColor);
    }else{
        double dzdx = w.x/(-d*k);
        double dzdy = w.y/(-d*k);
        //zi is the 1/z value of the current pixel
        double zi;
        for(int yi = yMin; yi <= yMax; yi++){
            xlAB = xlAC = xlBC = posInf;
            xrAB = xrAC = xrBC = negInf;
            if((yi - a2D.y)*(yi - b2D.y) <= 0 and a2D.y != b2D.y){
                xlAB = xrAB = a2D.x + (b2D.x - a2D.x)*((yi - a2D.y)/(b2D.y - a2D.y));
            }
            if((yi - a2D.y)*(yi - c2D.y) <= 0 and a2D.y != c2D.y){
                xlAC = xrAC = a2D.x + (c2D.x - a2D.x)*((yi - a2D.y)/(c2D.y - a2D.y));
            }
            if((yi - b2D.y)*(yi - c2D.y) <= 0 and b2D.y != c2D.y){
                xlBC = xrBC = b2D.x + (c2D.x - b2D.x)*((yi - b2D.y)/(c2D.y - b2D.y));
            }
            xl = roundToInt(std::min(std::min(xlAB, xlAC), xlBC) + 0.5);
            xr = roundToInt(std::max(std::max(xrAB, xrAC), xrBC) - 0.5);
            for(int x = xl; x <= xr; x++){
                zi = 1.0001*zg + (x - xg)*dzdx + (yi - yg)*dzdy;
                ColorDouble diffuseSpecularLight = totalColor;
                if(useLight){
                    double ziEye = 1.0/zi;
                    double coordEyeX = (x - dx) * (-ziEye) / d;
                    double coordEyeY = (yi - dy) * (-ziEye) / d;
                    Vector3D point = Vector3D::point(coordEyeX, coordEyeY, ziEye);
                    double cos;
                    Vector3D l;
                    for(Light &light: pointLights){
                        l = Vector3D::normalise(light.vector - point);
                        cos = l.dot(n);
                        if(cos > 0) {
                            red = (diffuseReflection.red * light.diffuseLight.red) * cos;
                            green = (diffuseReflection.green * light.diffuseLight.green) * cos;
                            blue = (diffuseReflection.blue * light.diffuseLight.blue) * cos;
                        }else{
                            red = 0.0;
                            green = 0.0;
                            blue = 0.0;
                        }
                        diffuseSpecularLight.red = (red + diffuseSpecularLight.red) < 1.0 ?
                                                   red + diffuseSpecularLight.red : 1.0;
                        diffuseSpecularLight.green = (green + diffuseSpecularLight.green) < 1.0 ?
                                                     green + diffuseSpecularLight.green : 1.0;
                        diffuseSpecularLight.blue = (blue + diffuseSpecularLight.blue) < 1.0 ?
                                                    blue + diffuseSpecularLight.blue : 1.0;
                    }
                    for(Light &light: specularLights){
                        if(light.infinite){
                            l = Vector3D::normalise(-light.vector);
                        }else{
                            l = Vector3D::normalise(light.vector - point);
                        }
                        cos = l.dot(n);
                        Vector3D r = Vector3D::normalise((2 * cos * n) - l);
                        Vector3D camera = Vector3D::normalise(Vector3D::point(0.0,0.0,0.0) - point);
                        double beta = std::pow(camera.dot(r), reflectionCoefficient);
                        if(cos > 0 and beta > 0){
                            red = (specularReflection.red * light.specularLight.red) * beta;
                            green = (specularReflection.green * light.specularLight.green) * beta;
                            blue = (specularReflection.blue * light.specularLight.blue) * beta;
                        }else{
                            red = 0.0;
                            green = 0.0;
                            blue = 0.0;
                        }
                        diffuseSpecularLight.red = (red + diffuseSpecularLight.red) < 1.0 ?
                                                   red + diffuseSpecularLight.red : 1.0;
                        diffuseSpecularLight.green = (green + diffuseSpecularLight.green) < 1.0 ?
                                                     green + diffuseSpecularLight.green : 1.0;
                        diffuseSpecularLight.blue = (blue + diffuseSpecularLight.blue) < 1.0 ?
                                                    blue + diffuseSpecularLight.blue : 1.0;
                    }
                    currentColor = img::Color(diffuseSpecularLight.red*255, diffuseSpecularLight.green*255, diffuseSpecularLight.blue*255);
                }
                if(zi < zBuf(x, yi)){
                    zBuf(x, yi) = zi;
                    image(x, yi) = currentColor;
                }
            }
        }
    }
}

img::EasyImage draw2DLines(Lines2D &lines, const int size, std::vector<double> bgc,
                           bool zBuffer = false, bool useLight = false, Lights3D lights = {},
                           std::vector<Figure> figures = {}){
    double xmin = 0.0;
    double xmax = 0.0;
    double ymin = 0.0;
    double ymax = 0.0;
    bool firstLine = false;

    for(Line2D &l: lines){
        double currentXmin = std::min(l.p1.x, l.p2.x);
        double currentXmax = std::max(l.p1.x, l.p2.x);
        double currentYmin = std::min(l.p1.y, l.p2.y);
        double currentYmax = std::max(l.p1.y, l.p2.y);

        //xmin, xmax, ymin and ymax have to be set to the first coordinates encountered
        if(!firstLine){
            xmin = currentXmin;
            xmax = currentXmax;
            ymin = currentYmin;
            ymax = currentYmax;
            firstLine = true;
        }
        if(currentXmin < xmin){xmin = currentXmin;}
        if(currentXmax > xmax){xmax = currentXmax;}
        if(currentYmin < ymin){ymin = currentYmin;}
        if(currentYmax > ymax){ymax = currentYmax;}
    }
    double xRange = xmax - xmin;
    double yRange = ymax - ymin;
    double imageX = size * (xRange/std::max(xRange, yRange));
    double imageY = size * (yRange/std::max(xRange, yRange));
    double d = 0.95 * imageX / xRange;
    double DCX = d * (xmin + xmax) / 2;
    double DCY = d * (ymin + ymax) / 2;
    double dX = (imageX / 2) - DCX;
    double dY = (imageY / 2) - DCY;

    img::EasyImage image(roundToInt(imageX), roundToInt(imageY), img::Color(
        bgc[0] * 255, bgc[1] * 255, bgc[2] * 255));

    if(figures.size() > 0){
        //z-buffering with triangles
        ZBuffer zBuf(roundToInt(imageX), roundToInt(imageY));
        for(Figure &figure: figures){
            for(Face &face: figure.faces){
                Vector3D A = figure.points[face.point_indexes[0]];
                Vector3D B = figure.points[face.point_indexes[1]];
                Vector3D C = figure.points[face.point_indexes[2]];
                draw_zbuf_triag(zBuf, image, A, B, C, d, dX, dY,
                                figure.ambientReflection,
                                figure.diffuseReflection,
                                figure.specularReflection,
                                figure.reflectionCoefficient,
                                lights, useLight);
            }
        }
    }else if(zBuffer){
        //3D-line drawing with z-buffering
        ZBuffer zBuf(roundToInt(imageX), roundToInt(imageY));
        for(Line2D &l : lines) {
            int x1 = roundToInt((l.p1.x * d) + dX);
            int y1 = roundToInt((l.p1.y * d) + dY);
            int x2 = roundToInt((l.p2.x * d) + dX);
            int y2 = roundToInt((l.p2.y * d) + dY);
            draw_zbuf_line(zBuf, image, x1, y1, l.z1, x2, y2, l.z2,
                           img::Color(l.color.blue * 255, l.color.green * 255, l.color.red * 255));
        }
    }else{
        //regular 2D line drawing
        for (Line2D &l : lines) {
            int x1 = roundToInt((l.p1.x * d) + dX);
            int y1 = roundToInt((l.p1.y * d) + dY);
            int x2 = roundToInt((l.p2.x * d) + dX);
            int y2 = roundToInt((l.p2.y * d) + dY);
            image.draw_line(x1, y1, x2, y2,
                            img::Color(l.color.blue * 255, l.color.green * 255, l.color.red * 255));
        }
    }
    return image;
}

std::vector<double> getAngledPoint(Point2D &p1, double angle){
    double angleRad = angle * M_PI / 180;

    double x2 = p1.x + std::cos(angleRad);
    double y2 = p1.y + std::sin(angleRad);

    return {x2, y2};
}

void drawLSystem(const LParser::LSystem2D &l_system, ColorDouble &color, Lines2D &lSystemLines){
    std::set<char> alphabet = l_system.get_alphabet();
    double angle = l_system.get_angle();
    unsigned int nr_iterations = l_system.get_nr_iterations();
    std::stack<Point2D> pointStack;
    std::stack<double> angleStack;

    std::string systemString = l_system.get_initiator();
    for(unsigned int i = 0; i <= nr_iterations; i++){
        std::string currentString = "";
        for(char c: systemString){
            if(alphabet.find(c) != alphabet.end()){
                currentString += l_system.get_replacement(c);
            }else{
                currentString += c;
            }
        }
        systemString = currentString;
    }

    Point2D currentPoint = Point2D();
    Point2D nextPoint;
    currentPoint.x = 0.0;
    currentPoint.y = 0.0;
    double currentAngle = l_system.get_starting_angle();

    for(char c: systemString){
        if(alphabet.find(c) != alphabet.end()){
            std::vector<double> angledPoints = getAngledPoint(currentPoint, currentAngle);
            if(l_system.draw(c)){
                nextPoint = Point2D();
                nextPoint.x = angledPoints[0];
                nextPoint.y = angledPoints[1];

                Line2D l = Line2D();
                l.p1 = currentPoint;
                l.p2 = nextPoint;
                l.color = color;
                lSystemLines.push_back(l);
            }
            currentPoint = Point2D();
            currentPoint.x = angledPoints[0];
            currentPoint.y = angledPoints[1];

        }else if(c == '+'){
            currentAngle = (currentAngle + angle);
        }else if(c == '-'){
            currentAngle = (currentAngle - angle);
        }else if(c == '('){
            Point2D savedPoint;
            savedPoint.x = currentPoint.x;
            savedPoint.y = currentPoint.y;
            pointStack.push(savedPoint);

            double currentAngleCopy = currentAngle;
            angleStack.push(currentAngleCopy);
        }else if(c == ')'){
            Point2D savedPoint = pointStack.top();
            pointStack.pop();
            currentPoint.x = savedPoint.x;
            currentPoint.y = savedPoint.y;
            currentAngle = angleStack.top();
            angleStack.pop();
        }
    }
}

//======================================================================================================//
//===============================================GENERATE IMAGE=========================================//

img::EasyImage generate_image(const ini::Configuration &configuration){
    // These variables are created regardless of the type
    std::string type = configuration["General"]["type"];
    int size = configuration["General"]["size"];
    std::vector<double> bgc = configuration["General"]["backgroundcolor"];

    if(type == "2DLSystem") {
        std::string file = configuration["2DLSystem"]["inputfile"];
        std::vector<double> lineRGB = configuration["2DLSystem"]["color"];

        ColorDouble color(lineRGB[0], lineRGB[1], lineRGB[2]);

        LParser::LSystem2D l_system;
        std::ifstream input_stream(file);
        input_stream >> l_system;
        input_stream.close();

        Lines2D lines;
        drawLSystem(l_system, color, lines);


        return draw2DLines(lines, size, bgc);
    }else if(type == "Wireframe" or type == "ZBufferedWireframe" or type == "ZBuffering"
             or type == "LightedZBuffering"){
        bool zBuffering = (type == "ZBufferedWireframe");
        int nrFigures = configuration["General"]["nrFigures"];
        std::vector<double> eye = configuration["General"]["eye"];
        Vector3D eyePoint;
        eyePoint.x = eye[0];
        eyePoint.y = eye[1];
        eyePoint.z = eye[2];
        Matrix eyePMatrix = eyePointTrans(eyePoint);

        //create all Light objects
        bool useLights = (type == "LightedZBuffering");
        Lights3D lights;
        if(!useLights){
            Light light;
            light.ambientLight = ColorDouble(1.0, 1.0, 1.0);
            light.diffuseLight = ColorDouble(0.0, 0.0, 0.0);
            light.specularLight = ColorDouble(0.0, 0.0, 0.0);
            lights = {light};
        }else{
            int nrLights = configuration["General"]["nrLights"];
            std::vector<double> ambientLight, diffuseLight, specularLight;
            for (int i = 0; i < nrLights; i++) {
                std::string lightString = "Light" + std::to_string(i);
                Light currentLight;
                bool infinite;
                try{
                    infinite = configuration[lightString]["infinity"];
                    currentLight.infinite = infinite;
                    if(infinite){
                        std::vector<double> direction = configuration[lightString]["direction"];
                        currentLight.vector = Vector3D::vector(direction[0],direction[1],direction[2]) * eyePMatrix;
                    }else{
                        std::vector<double> location = configuration[lightString]["location"];
                        currentLight.vector = Vector3D::point(location[0],location[1],location[2]) * eyePMatrix;
                    }
                }catch (ini::NonexistentEntry){}
                try{
                    ambientLight = configuration[lightString]["ambientLight"];
                    currentLight.ambientLight = ColorDouble(ambientLight[0], ambientLight[1], ambientLight[2]);
                }catch (ini::NonexistentEntry){}
                try{
                    diffuseLight = configuration[lightString]["diffuseLight"];
                    currentLight.diffuseLight = ColorDouble(diffuseLight[0], diffuseLight[1], diffuseLight[2]);
                }catch (ini::NonexistentEntry){}
                try{
                    specularLight = configuration[lightString]["specularLight"];
                    currentLight.specularLight = ColorDouble(specularLight[0], specularLight[1], specularLight[2]);
                    currentLight.specular = true;
                }catch (ini::NonexistentEntry){}
                lights.push_back(currentLight);
            }
        }

        Figures3D figures;

        for(int i = 0; i < nrFigures; i++){
            std::string figureString = "Figure" + std::to_string(i);

            Figure currentFigure;
            //this vector is used in case of fractal figures/menger sponge
            Figures3D fractalVector = {};
            std::string figureType = configuration[figureString]["type"];
            double scale = configuration[figureString]["scale"];
            double xAngle = configuration[figureString]["rotateX"];
            double yAngle = configuration[figureString]["rotateY"];
            double zAngle = configuration[figureString]["rotateZ"];
            std::vector<double> centerVec = configuration[figureString]["center"];
            Vector3D center = Vector3D::point(centerVec[0], centerVec[1], centerVec[2]);

            if(figureType == "LineDrawing"){
                int nrPoints = configuration[figureString]["nrPoints"];
                int nrLines = configuration[figureString]["nrLines"];

                for(int j = 0; j < nrPoints; j++){
                    std::string pointString = "point" + std::to_string(j);

                    std::vector<double> xyzPoint = configuration[figureString][pointString];
                    Vector3D point;
                    point.x = xyzPoint[0];
                    point.y = xyzPoint[1];
                    point.z = xyzPoint[2];
                    currentFigure.points.push_back(point);
                }
                for(int j = 0; j < nrLines; j++){
                    std::string lineString = "line" + std::to_string(j);
                    //Face is used as a line by only using two points
                    Face line;
                    std::vector<int> lines = configuration[figureString][lineString];
                    line.point_indexes = lines;
                    currentFigure.faces.push_back(line);
                }
            }else if(figureType == "Cube"){
                currentFigure = createCube();
            }else if(figureType == "Tetrahedron"){
                currentFigure = createTetrahedron();
            }else if(figureType == "Octahedron"){
                currentFigure = createOctahedron();
            }else if(figureType == "Icosahedron") {
                currentFigure = createIcosahedron();
            }else if(figureType == "Sphere"){
                int n = configuration[figureString]["n"];
                currentFigure = createSphere(1, n);
            }else if(figureType == "Dodecahedron") {
                currentFigure = createDodecadron();
            }else if(figureType == "Cone") {
                int n = configuration[figureString]["n"];
                double h = configuration[figureString]["height"];
                currentFigure = createCone(n, h);
            }else if(figureType == "Cylinder") {
                int n = configuration[figureString]["n"];
                double h = configuration[figureString]["height"];
                currentFigure = createCylinder(n, h);
            }else if(figureType == "Torus") {
                double r = configuration[figureString]["r"];
                double R = configuration[figureString]["R"];
                int n = configuration[figureString]["n"];
                int m = configuration[figureString]["m"];
                currentFigure = createTorus(r, R, n, m);
            }else if(figureType == "BuckyBall"){
                currentFigure = createBuckyball();
            }else if(figureType == "MengerSponge"){
                int nr_iterations = configuration[figureString]["nrIterations"];
                createSponge(nr_iterations, fractalVector);
            }else if(figureType == "FractalCube"){
                currentFigure = createCube();
                double fractalScale = configuration[figureString]["fractalScale"];
                int nrIterations = configuration[figureString]["nrIterations"];
                generateFractal(currentFigure, fractalVector, nrIterations, fractalScale);
            }else if(figureType == "FractalTetrahedron"){
                currentFigure = createTetrahedron();
                double fractalScale = configuration[figureString]["fractalScale"];
                int nrIterations = configuration[figureString]["nrIterations"];
                generateFractal(currentFigure, fractalVector, nrIterations, fractalScale);
            }else if(figureType == "FractalOctahedron"){
                currentFigure = createOctahedron();
                double fractalScale = configuration[figureString]["fractalScale"];
                int nrIterations = configuration[figureString]["nrIterations"];
                generateFractal(currentFigure, fractalVector, nrIterations, fractalScale);
            }else if(figureType == "FractalIcosahedron") {
                currentFigure = createIcosahedron();
                double fractalScale = configuration[figureString]["fractalScale"];
                int nrIterations = configuration[figureString]["nrIterations"];
                generateFractal(currentFigure, fractalVector, nrIterations, fractalScale);
            }else if(figureType == "FractalBuckyBall"){
                currentFigure = createBuckyball();
                double fractalScale = configuration[figureString]["fractalScale"];
                int nrIterations = configuration[figureString]["nrIterations"];
                generateFractal(currentFigure, fractalVector, nrIterations, fractalScale);
            }else if(figureType == "FractalDodecahedron") {
                currentFigure = createDodecadron();
                double fractalScale = configuration[figureString]["fractalScale"];
                int nrIterations = configuration[figureString]["nrIterations"];
                generateFractal(currentFigure, fractalVector, nrIterations, fractalScale);
            }else{
                std::cerr << figureType << " is an invalid type\n";
                continue;
            }

            //set color
            ColorDouble color, ambientRef, diffuseRef, specularRef;
            bool useSpecular = false;
            int reflectionCoefficient;
            if(!useLights){
                std::vector<double> lineRGB = configuration[figureString]["color"];
                color = ColorDouble(lineRGB[0], lineRGB[1], lineRGB[2]);
            }else{
                try{
                    std::vector<double> ambient = configuration[figureString]["ambientReflection"];
                    ambientRef = ColorDouble(ambient[0],ambient[1],ambient[2]);
                }catch (ini::NonexistentEntry){}
                try{
                    std::vector<double> diffuse = configuration[figureString]["diffuseReflection"];
                    diffuseRef = ColorDouble(diffuse[0],diffuse[1],diffuse[2]);
                }catch (ini::NonexistentEntry){}
                try{
                    std::vector<double> specular = configuration[figureString]["specularReflection"];
                    reflectionCoefficient = configuration[figureString]["reflectionCoefficient"];
                    specularRef = ColorDouble(specular[0],specular[1],specular[2]);
                    useSpecular = true;
                }catch (ini::NonexistentEntry){}
            }

            //apply all transformations
            Matrix sM = scaleFigure(scale);
            Matrix rX = rotateX(xAngle);
            Matrix rY = rotateY(yAngle);
            Matrix rZ = rotateZ(zAngle);
            Matrix tM = translate(center);
            Matrix transformationMatrix = sM*rX*rY*rZ*tM;
            if(fractalVector.size() > 0){
                for(Figure &fig: fractalVector){
                    if(!useLights){fig.ambientReflection = color;}
                    else{
                        fig.ambientReflection = ambientRef;
                        fig.diffuseReflection = diffuseRef;
                        fig.specularReflection = specularRef;
                        if(useSpecular){fig.reflectionCoefficient = reflectionCoefficient;}
                    }
                    applyTransformation(fig, transformationMatrix);
                    figures.push_back(fig);
                }
            }else{
                if(!useLights){currentFigure.ambientReflection = color;}
                else{
                    currentFigure.ambientReflection = ambientRef;
                    currentFigure.diffuseReflection = diffuseRef;
                    currentFigure.specularReflection = specularRef;
                    if(useSpecular){currentFigure.reflectionCoefficient = reflectionCoefficient;}
                }
                applyTransformation(currentFigure, transformationMatrix);
                figures.push_back(currentFigure);
            }
        }

        applyTransformation(figures, eyePMatrix);
        Lines2D lines = doProjection(figures, 1);

        if(type == "ZBuffering" or type == "LightedZBuffering"){
            for(int i = 0; i < figures.size(); i++){
                triangulateFigure(figures[i]);
            }
            return draw2DLines(lines, size, bgc, zBuffering, useLights, lights, figures);
        }else{
            return draw2DLines(lines, size, bgc, zBuffering);
        }

    }else{
        std::cerr << type << " is an invalid type\n";
    }
}

int main(int argc, char const* argv[])
{
        int retVal = 0;
        try
        {
                for(int i = 1; i < argc; ++i)
                {
                        ini::Configuration conf;
                        try
                        {
                                std::ifstream fin(argv[i]);
                                fin >> conf;
                                fin.close();
                        }
                        catch(ini::ParseException& ex)
                        {
                                std::cerr << "Error parsing file: " << argv[i] << ": " << ex.what() << std::endl;
                                retVal = 1;
                                continue;
                        }

                        img::EasyImage image = generate_image(conf);
                        if(image.get_height() > 0 && image.get_width() > 0)
                        {
                                std::string fileName(argv[i]);
                                std::string::size_type pos = fileName.rfind('.');
                                if(pos == std::string::npos)
                                {
                                        //filename does not contain a '.' --> append a '.bmp' suffix
                                        fileName += ".bmp";
                                }
                                else
                                {
                                        fileName = fileName.substr(0,pos) + ".bmp";
                                }
                                try
                                {
                                        std::ofstream f_out(fileName.c_str(),std::ios::trunc | std::ios::out | std::ios::binary);
                                        f_out << image;

                                }
                                catch(std::exception& ex)
                                {
                                        std::cerr << "Failed to write image to file: " << ex.what() << std::endl;
                                        retVal = 1;
                                }
                        }
                        else
                        {
                                std::cout << "Could not generate image for " << argv[i] << std::endl;
                        }
                }
        }
        catch(const std::bad_alloc &exception)
        {
    		//When you run out of memory this exception is thrown. When this happens the return value of the program MUST be '100'.
    		//Basically this return value tells our automated test scripts to run your engine on a pc with more memory.
    		//(Unless of course you are already consuming the maximum allowed amount of memory)
    		//If your engine does NOT adhere to this requirement you risk losing points because then our scripts will
		//mark the test as failed while in reality it just needed a bit more memory
                std::cerr << "Error: insufficient memory" << std::endl;
                retVal = 100;
        }
        return retVal;
}
