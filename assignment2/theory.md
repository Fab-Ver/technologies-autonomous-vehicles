# Lane Detection

Lane detection is crucial for driver assistance systems and for autonomous vehicles, it is used to compute the position of the vehicle wrt to the lane lines. 

The information can be used to: 

- Warn the drier in case of potential involuntary lane crossing.
- Control the steering to maintain a desired position wrt the lane lines
- Compute the radius of curvature of the lane

## Image Pipeline

![image.png](attachment:2c50f12d-3949-4254-b840-4c844195ad4a:image.png)

The camera calibration must be performed every time the camera is moved. 

A camera is a mapping between the 3D world and a 2D image. Camera can be modelled using matrices with properties that represents the camera 3D to 3D mapping. 

The basic model is the pinhole model. 

### Pinhole Camera Model

We consider the **central projection** of points in space onto a plane

- **Centre of projection** = origin of a Euclidean coordinate system
- **Plane** $Z = f$ is called the **image plane** or **focal plane**

A point in space $\mathbf{X} = (X, Y, Z)^\top$ is mapped to the point on the image plane where a line joining the point $\mathbf{X}$ to the centre $\mathbf{C}$ of projection meets the image plane

- The point $(X, Y, Z)^\top$ is mapped to the point $(fX/Z,\ fY/Z,\ f)^\top$ on the image plane
- The mapping from Euclidean 3-space to Euclidean 2-space is (by ignoring the third image coordinate):

$$
(X,\ Y,\ Z)^\top \mapsto (fX/Z,\ fY/Z)^\top
$$

![image.png](attachment:8d7459ad-67f8-4651-9ecb-2594398e730a:image.png)

---

### Homogeneous Coordinates

**Homogeneous coordinates** are a projective geometry representation that extends ordinary (Euclidean) coordinates so that:

- Translations become **linear operations**
- **Perspective projections** can be expressed as matrix multiplication
- **Points at infinity** can be represented

Example: 

- Euclidean point: $(x_e,\ y_e)$
- Corresponding homogeneous coordinates: $(x,\ y,\ w)$
- Conversions:
    
    $$
    x_e = \frac{x}{w}, \quad y_e = \frac{y}{w}
    $$
    
- Usually $w = 1$, so we have: $(x, y) \rightarrow (x,\ y,\ 1)$

Translation

- We want to express the translation of the point $(x, y)$ to the point $(x^t = x + t_x,\ y^t = y + t_y)$
- Using homogeneous coordinates, the operation becomes a **matrix multiplication**:

$$

\begin{bmatrix} x^t \\ y^t \\ 1 \end{bmatrix}=\begin{bmatrix} 1 & & t_x \\ & 1 & t_y \\ & & 1 \end{bmatrix}
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}

$$

Rotation: we want to express the rotation using rotation matrix $R$ of the point $(x, y, z)$:

$$

\begin{bmatrix} x^R \\ y^R \\ z^R \\ 1 \end{bmatrix}=\begin{bmatrix} R & 0 \\ 0 & 1 \end{bmatrix}
\begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}
$$

---

### Pinhole Camera Model

A model of a camera can be built starting from its **characteristics**

**Assumptions:**

- The camera center is defined as $C$
- The image plane is located at distance $f$ (**focal length**) from the pinhole
- We have a **camera coordinate system** where:
    - The origin is the camera center
    - The $z$ axis points forward
    - The image plane is perpendicular to the $z$ axis
- The 3D world has its origin in $O$
- The projection of the 3D world point is **rotated** using rotation matrix $R$ and **translated** by $t$

![image.png](attachment:0ce1fd9f-21ad-47e8-bf9f-776d492951a6:image.png)

---

**3D world point** in the 3D world reference system:

$$
\tilde{X}_w = \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}

$$

**3D world point** in the camera reference system:

$$

\tilde{X}_c = \begin{bmatrix} X_c \\ Y_c \\ Z_c \\ 1 \end{bmatrix}

$$

**Image point:**

$$
\tilde{x} = \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

---

The transformation from 3D world coordinates to camera coordinates is $X_c = RX_w + t$ that in homogeneous coordinates this becomes:

$$
\tilde{X}_c = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix} \tilde{X}_w
$$

The **image plane** is defined by:

- **Focal length** in pixels: $(f_x,\ f_y)$
- **Principal point** (center of the sensor) in pixels: $(c_x,\ c_y)$

The **intrinsic calibration matrix** $K$ conveys the image characteristics:

$$

K = \begin{bmatrix} f_x & & c_x \\ & f_y & c_y \\ & & 1 \end{bmatrix}

$$

> **Note:** The intrinsic matrix is **independent** of the position of the camera in the space (vehicle); it depends on the **physical properties of the sensor**.
> 

The projection of the 3D point in the camera reference system onto the image plane:

$$
\tilde{x} = K \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}

$$

Full Pipeline (Putting It All Together):

$$

\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}=\begin{bmatrix} f_x & & c_x \\ & f_y & c_y \\ & & 1 \end{bmatrix}
\begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}
\begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}

$$

---

The equation is normally written as:

$$
\tilde{x} = K[R\ \ t]\tilde{X}_w
$$

- $P = K[R\ \ t]$ is the **pinhole camera model**
- $K$ is the **intrinsic (internal) camera calibration matrix**, that defines the **properties of the camera**
- $[R\ \ t]$ is the **extrinsic (external) calibration matrix**, that defines the **location of the camera in the vehicle**

### Camera Distortion

![image.png](attachment:6ea70f21-edd2-4afb-bb41-c851926915f1:image.png)

The pinhole model is described as *ideal* precisely because it does not account for any optical distortion. In reality, lenses — especially the cheap ones typically used in automotive applications — introduce distortions that cause straight lines in the world to appear curved in the image. Two common types are **barrel distortion**, where the image appears to bulge outward from the center, and **pincushion distortion**, where it pinches inward.

![image.png](attachment:54e5a070-5933-40ce-aca9-94814070535e:image.png)

To correct for **radial distortion**, the following correction equations are applied:

$$
\hat{x} = x(1 + \kappa_1(x^2 + y^2) + \kappa_2(x^2 + y^2)^2)
\\\hat{y} = y(1 + \kappa_1(x^2 + y^2) + \kappa_2(x^2 + y^2)^2)
$$

where $k_1$ and $k_2$ are the **radial distortion parameters**.

> **Note:** The lenses used in automotive applications are usually very cheap and do not correct these distortions.
> 

---

### Camera Calibration

**Camera calibration** consists in calculating the internal and external parameters for a specific camera, including the distortion parameters. 

The procedure unfolds as a sequence of steps: first, a pattern (normally a **checkerboard**) is printed and attached to a planar surface, and a few images of it are taken under different orientations by moving either the plane or the camera. Feature points are then detected in the images, and the intrinsic and extrinsic parameters are estimated using the **closed-form solution**. After that, the coefficients of the radial distortion are estimated by solving a linear least-squares problem, and finally all parameters, including lens distortion parameters, are refined by a minimization procedure.

---

#### Closed Form Solution for Parameter Estimation

Considering a feature point in the world reference frame and its mapped point in the camera reference frame, the relationship is:

$$
s \begin{pmatrix} u \\ v \\ 1 \end{pmatrix} = K[R \mid t] \begin{pmatrix} X \\ Y \\ Z \\ 1 \end{pmatrix}
$$

where $s$ is a **scaling factor**, $(u, v)$ are the **camera reference frame** coordinates, and $(X, Y, Z, 1)^\top$ is the point in the **world reference frame**. Assuming the calibration plane is such that $Z = 0$, and setting $R = [r_1\ r_2\ r_3]$, the equation simplifies to:

$$
s \begin{pmatrix} u \ v \ 1 \end{pmatrix} = K[r_1\ r_2\ r_3 \mid t] \begin{pmatrix} X \\ Y \\ 0 \\ 1 \end{pmatrix} = \lambda K[r_1\ r_2 \mid t] \begin{pmatrix} X \\ Y \\ 1 \end{pmatrix}
$$

We then define $\lambda K[r_1\ r_2\ t] = H = [h_1\ h_2\ h_3]$, the **homography matrix**. By exploiting the fact that $r_1$ and $r_2$ are orthogonal, two constraints arise:

$$
\begin{cases} h_1^T K^{-T} K^{-1} h_2 = 0 \\ h_1^T K^{-T} K^{-1} h_1 = h_2^T K^{-T} K^{-1} h_2 \end{cases}
$$

The system has **8 degrees of freedom** (combining camera internal and external parameters), and by using at least two independent images, it is possible to compute the camera's internal parameters $K$.

Once $K$ is known, the rotation and translation can be recovered as:

$$
r_1 = \lambda K^{-1} h_1 \\ r_2 = \lambda K^{-1} h_2 \\ r_3 = r_1 r_2 \\ t = \lambda K^{-1} h_3
$$

where the scalar $\lambda$ is defined as:

$$
\lambda = \frac{1}{|K^{-1} h_1|} = \frac{1}{|K^{-1} h_2|}
$$

---

#### Estimate the Coefficients of the Radial Distortion

The **radial distortion parameters** $(k_1, k_2)$ can be estimated by solving $2nm$ equations, where each ideal pixel position $(u, v)$ is computed starting from the actual pixel position $(x, y)$, while considering $m$ points in $n$ images. The linear system takes the form:

$$
\begin{bmatrix} (u - u_0)(x^2 + y^2) & (u - u_0)(x^2 + y^2)^2 \\ (v - v_0)(x^2 + y^2) & (v - v_0)(x^2 + y^2)^2 \end{bmatrix} \begin{bmatrix} k_1 \\ k_2 \end{bmatrix} = \begin{bmatrix} \breve{u} - u \\ \breve{v} - v \end{bmatrix}
$$

---

#### Refine All Parameters

Given $n$ images of a model plane and $m$ points on the model plane, the **maximum likelihood estimate** of the parameters is obtained by minimizing the following functional:

$$
\sum_{i=1}^{n} \sum_{j=1}^{m} | \mathbf{m}_{ij} - \hat{\mathbf{m}}(A,\ R_i,\ t_i,\ M_j) |^2
$$

where $\hat{\mathbf{m}}(A, R_i, t_i, M_j)$ denotes the **projection of point $M_j$ in image $i$** according to $\lambda K [r_1\ r_2 \mid t], M$.

---

#### Camera Calibration in MATLAB

In practice, calibration is performed using images of a checkerboard pattern. The example uses nine test images at a resolution of $1072 \times 712$ pixels, featuring a $7 \times 10$ checkerboard where each square is $29,\text{mm}$ wide.

![image.png](attachment:f943cb6e-a356-43be-8e49-6799450bdcfb:image.png)

The first step is to **load the images and detect the checkerboard feature points**, whose coordinates are expressed in the camera reference plane:

```matlab
images = imageSet(fullfile(toolboxdir('vision'),'visiondata','calibration','mono'));
imageFileNames = images.ImageLocation;
[imagePoints, boardSize] = detectCheckerboardPoints(imageFileNames);
```

Next, the **world reference coordinates** of the checkerboard points are generated and the calibration is performed:

```matlab
squareSizeInMM = 29;
worldPoints = generateCheckerboardPoints(boardSize, squareSizeInMM);

I = readimage(images, 1);
imageSize = [size(I, 1), size(I, 2)];
params = estimateCameraParameters(imagePoints, worldPoints, 'ImageSize', imageSize);
```

This operation returns both the **internal** and **external camera parameters**.

![Representation of external parameters](attachment:08d49cd8-d9ed-4643-94bc-c23bba388e6e:image.png)

Representation of external parameters

---

#### Parameter Estimation Error

Since parameter estimation is fundamentally an **optimization process**, a certain estimation error is always to be expected. The error varies across different images used in calibration, reflecting differences in viewing angle, distance, and pattern coverage.

![image.png](attachment:1775caed-df72-4850-b83d-f79a97116ebf:image.png)

#### Distortion Correction

After camera calibration the distortion affecting the image can be removed by applying the radial distortion parameters. 

![image.png](attachment:b1e71fb2-3492-4e6e-b44d-dbe5ef1644dd:image.png)

After camera calibration: 

```matlab
J1 = undistortImage(I,params);
```

## Perspective Transformation

### Why Perspective Transformation is Needed?

The image obtained by the car camera contains a large **non-road area**, such as sky, trees on roadside, etc. Processing the full image will **increase the computational complexity** and **reduce the real-time capability.** 

The invalid area will **interfere the lane information** and affect the **detection accuracy.** Only the area shown in the red box is the **Region of Interest (ROI)** for lane detection

![image.png](attachment:ddc43f25-6b3f-4e59-b88c-adb10bc585ae:image.png)

### Inverse Perspective Transformation

Due to the **imaging perspective effect**, the traffic lanes appear as two **non-parallel lines** in the original image

The **inverse perspective transformation** eliminates the perspective effect and produces a **top view (Bird's Eye View)** image that is consistent with the actual situation. 

In the resulting top view image, the lane lines are **vertical and parallel** to each other, which facilitates identification in subsequent algorithms. 

![image.png](attachment:f4cd477b-10b2-439c-b15d-f80ce2485c01:image.png)

### ROI Shape and Transformation Definition

The ROI should be converted into an **inverted trapezoidal shape** after the transformation

The **inverse perspective transformation** is the process that converts a **rectangle** into an **inverted trapezoid.** 

![image.png](attachment:0e405c39-502c-4295-8bd2-7f91e341b3fd:image.png)

### Mathematical Formulation

- $(u, v)$ denotes the **ROI coordinate system**
- $(x, y)$ denotes the **transformed top view coordinate system**

The mapping relationship can be expressed as:

$$
P = QM
$$

where:

- $P = [x' \quad y' \quad w']$
- $x = \dfrac{x'}{w'}, \quad y = \dfrac{y'}{w'}$
- $Q = [u \quad v \quad 1]$
- $M = \begin{bmatrix} a & d & g \\ b & e & h \\ c & f & 1 \end{bmatrix}$, the **transformation matrix** from the rectangular to the inverted trapezoidal

$M$ is a $3\times3$ homography matrix. The homogeneous coordinates $(x', y', w')$ are normalized by dividing by $w'$ to recover the Cartesian coordinates $(x, y)$ in the top-view plane.

### Perspective Transformation in MATLAB

Knowing the **internal and external camera parameters**, the steps are:

1. Create a `monoCamera` sensor object
2. Define the **ROI**
3. Compute the **inverse trapezoidal transformation** using the `birdsEyeView()` operator
4. Apply the transformation to the input image

```matlab
focalLength = [309.4362 344.2161];
principalPoint = [318.9034 257.5352];
imageSize = [480 640];
camIntrinsics = cameraIntrinsics(focalLength, principalPoint, imageSize);

height = 2.1798; pitch = 14;
sensor = monoCamera(camIntrinsics, height, 'Pitch', pitch);

distAhead = 30;
spaceToOneSide = 6;
bottomOffset = 3;
outView = [bottomOffset, distAhead, -spaceToOneSide, spaceToOneSide];

outImageSize = [NaN, 250];
birdsEye = birdsEyeView(sensor, outView, outImageSize);

I = imread('road.png');
BEV = transformImage(birdsEye, I);
```

Key parameters:

- `focalLength`: camera focal length in pixels $[f_x, f_y]$
- `principalPoint`: principal point (optical center) $[c_x, c_y]$
- `imageSize`: image resolution $[rows, cols]$
- `height`: camera height above ground (meters)
- `pitch`: camera pitch angle (degrees)
- `outView`: defines the ROI in world coordinates $[\text{bottomOffset}, \text{distAhead}, -\text{side}, +\text{side}]$
- `outImageSize`: output image size (`NaN` = auto-computed)

## Lane Detection

Lane detection consists of three main steps:

1. Grayscale conversion and noise reduction for lane enhancement
2. Image binarization
3. Extraction of lane characteristics.

---

### Grayscale Conversion and Noise Reduction

The image is first transformed to grayscale in order to reduce the amount of information that needs to be processed. However, random noise introduced by the camera can disturb the image and potentially compromise the results of lane detection. 

To address this, **Gaussian blur** is applied as a low-pass filter. It works by using a Gaussian function to calculate the transformation applied to each pixel in the image, defined as:

$$
G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}
$$

![image.png](attachment:a09efe78-c52a-449c-b02d-73fcbee248ad:image.png)

---

#### Convolution

The Gaussian blur filter is defined by a **convolution matrix** called the **kernel**, and it is applied to the image through matrix multiplication. The filtered image $g(x, y)$ is obtained as:

$$
g(x, y) = w \cdot f(x, y) = \sum_{s=-a}^{a} \sum_{t=-b}^{b} w(s, t) \cdot f(x - s,\ y - t)
$$

where $f(x, y)$ is the source image and $w$ is the kernel. The parameters $s$ and $t$ are selected according to a specific goal; for Gaussian blur, the kernel $w$ is either **3×3** or **5×5**.

The two standard Gaussian blur kernels are:

$$
w_{3\times3} = \frac{1}{16} \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix} \\w_{5\times5} = \frac{1}{256} \begin{bmatrix} 1 & 4 & 6 & 4 & 1 \\ 4 & 16 & 24 & 16 & 4 \\ 6 & 24 & 36 & 24 & 6 \\ 4 & 16 & 24 & 16 & 4 \\ 1 & 4 & 6 & 4 & 1 \end{bmatrix}
$$

![image.png](attachment:797948f9-9c94-44e2-89c3-beb80474da7a:image.png)

---

### Lane Enhancement

For efficient implementation, two **separable kernels** are used instead of a single 2D kernel. The filter is decomposed into one kernel along $x$ and one along $y$:

$$
g(x) = \frac{1}{\sigma_x^2} e^{\left(\frac{-x^2}{2\sigma_x^2}\left(1 - \frac{x^2}{\sigma_x^2}\right)\right)}
$$

$$
g(y) = e^{\left(\frac{-1}{2\sigma_y^2} y^2\right)}
$$

---

### Image Binarization

To distinguish lane lines from the background, the image is **binarized** based on the grayscale value of each pixel. Each pixel is assigned either 255 (white) or 0 (black) depending on whether its grayscale value meets a threshold $Th$:

$$
b(x, y) = \begin{cases} 255 & \Rightarrow g(x, y) \geq Th \\ 0 & \Rightarrow g(x, y) < Th \end{cases}
$$

The critical challenge is selecting the right threshold $Th$. Since illumination varies significantly across different road scenes, a fixed threshold cannot reliably separate lane lines from the background. Instead, an **iterative method** is used to determine an optimal, adaptive threshold.

The process begins by setting an initial threshold:

$$
Th_0 = \frac{g_{max} + g_{min}}{2}
$$

where $g_{max}$ and $g_{min}$ are the maximum and minimum gray values in the image. The image is then divided into two regions, $A$ (pixels above threshold) and $B$ (pixels below), and their average gray values are computed:

$$
g_A = \frac{\sum_{g(x,y) \geq Th_i} g(x,y)}{\sum_{g(x,y) \geq Th_i} 1} \\ g_B = \frac{\sum_{g(x,y) < Th_i} g(x,y)}{\sum_{g(x,y) < Th_i} 1}
$$

The threshold is then updated as:

$$
Th_{i+1} = \frac{g_A + g_B}{2}
$$

This process repeats until convergence, i.e., until $Th_{i+1} = Th_i$. The resulting threshold is capable of adapting to varied illumination conditions.

---

### Extraction of Lane Characteristics

In the binary image, pixels corresponding to lane lines have a value of **255**, while all other pixels are **0**. To locate the lanes, the pixel values in each column of the image are summed, building a **histogram** of pixel counts across horizontal positions. The two peaks in this histogram correspond to the positions of the **lane delimiters**.

![image.png](attachment:977ad3d0-83de-4372-bd9b-8551384f96b3:image.png)