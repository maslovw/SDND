# **Finding Lane Lines on the Road** 


### 1. Pipeline description

First, the image is normalized by brightness, so the shadows make less impact

Then, I crop the image in trapezoid shape and apply color filtering.

I noticed that cv2.threshold method gives good result at this point, especially 
with the challenge video, where we have a very bright spot on the road.

After applying threshold I convert the image into grayscale and apply blurring filter

Next step would be Canny edge detection with cv2.dilate method, that makes lines
more solid.

At the end I apply cv2.HoughLinesP, which can still have a lot of noise, 
so I added lines `filtering`: for each line in the output of HoughLines 
I find k and b values (y = kx + b), where k is the line slope and b is 
the vertical intercept of the given line. This allows me to calculate 
top and bottom points of the line. 

After finding all the necessary values I divide the array of lines by the 
slop: right lines have k>0, left lines have k<0. All the lines with the 
abs(slop) < 0.3 would be ignored as too close to horizontal line.

Finally I reject outliners in both arrays (left and right lines) using this
method: `data[abs(data - data.mean()) < m * data.std()]`, by k parameter and 
then by x_bottom. Then I calculate the mean line of all, finding the middle.

This line should represent the lane.

I added the filtering method to hough_lines method, so it's just easier to 
use.

Here's are some examples:

![challenge01.jpg][test_images_output/challenge01.jpg]
![challenge03.jpg][test_images_output/challenge03.jpg]
![solidWhiteCurve.jpg][test_images_output/solidWhiteCurve.jpg]
![solidYellowCurve.jpg][test_images_output/solidYellowCurve.jpg]


### 2. Potential shortcomings with my current pipeline


One potential shortcoming would be, and it still happens when there is a 
different type of material, that road is made of, and the brightness of it 
is quite different. I tried to filter it, quite successfully, comparing with
my first results, but still seems unstable, and I'm sure it won't survive in
a different test set.

Another shortcomings could be the road crossing, another car, that is too close
or any other object (maybe pothole) on the road.

I have no clue what will happen with road repairing situation, when we can have
yellow temporary lanes, that have priority, and we shouldn't care about white lanes.

Another problem would be night test set... big light gradient, I don't think my 
algorithm will survive that.



![challenge02.jpg][test_images_output/challenge02.jpg]
![challenge04.jpg][test_images_output/challenge04.jpg]

### 3. Possible improvements to my pipeline

Possible improvement would be to implement a method that will sort lines by the 
groups (by parameters k and x_bottom), and calculating the distance between left
and right lanes, to find the right group if lines. 

I suppose there are some other image processing methods, that I could use.


