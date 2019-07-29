			*************************************** 
				      - README -
			*************************************** 

*****************
I) INTRODUCTION *
*****************

This Readme file is here to explain what this project is about and how to proceed in order to run the application. We also mention a few things to keep in mind.



****************
II) MOTIVATION *
****************

There are already a few interactive web applications to visually explore the K-means algorithm. I have developed this one with a clear goal in mind : I wanted to expose the strengths and shortcomings of K-Means and not only provide a description for it. On top of being able to interactively advance from one step to the next in the learning procedure, I made sure to provide the user with enough control over some of the most influential aspects, such as the nature and shape of the dataset and initialization parameters of the algorithm. I have also added two more graphs to give the user more feedback about the current setting. This will hopefully allow for a better understanding of the practical issues that arise when dealing with K-means and clustering in general.



************
III) SETUP *
************

The first thing to do is to download the entire folder containing the application from the google drive, and store it somewhere on your local drive.

Next, and before attempting to execute the program, you should read the Requirements.txt file available in this folder. It contains the Python packages (and their versions) required to run the application. All of those packages can be installed via anaconda (either through the anaconda-navigator or conda install), or using pip install.



***************
IV) EXECUTION *
***************

Once it is done, you're all set and all that is left is to run the application.

In order to do so, you need to execute the python file named Interactive-kmeans-app.py
You can do that either from an IDE such as Spyder (by loading the script and pressing Run, duh ...) or from the Command-Line. If you do not know how to execute a python file from the command-line, head over here : https://www.pythoncentral.io/execute-python-script-file-shell/

Once you run the script, you will need to go to your browser and enter the address : http://127.0.0.1:8050/
For those of you that are familiar with computer networking, the Web Application is being run in local mode. And the 127.0.0.1 is the conventionally used IP address for localhost (loopback address). 8050 is the port number. Buuuuut, I digress ...

Finally, enjoy your web app !

Remark : you do not need to read the next section of this file unless you want to know some details about the implementation. If all you're here for is to experiment with the app, then you can already head over to your browser !



**************************
V) PROGRAM SPECIFICATION *
**************************

	A) Web Application

The web application was developed using the package Dash. It's a newly developed framework (it is still under development), that aims to make it easier to build Analytic Web Applications in python. The focus here is on Analytic as this is a specialized package and not something you'd use for any web application. It is somewhat equivalent to Shiny in R (but more flexible). This project belongs to Plotly and is based on the Plotly visualization package. If reporting findings through visualization is something you're keen on, then you should know that building analytic web applications is one of the best ways to deliver results and to enhance your portfolio/resume. Thus, I suggest you put some time into learning how to use this framework as it's gaining a lot of traction.

	B) Algorithm

The main algorithm of this program is obviously K-Means. In order to be able to store the intermediate steps of a K-means fitting procedure (which is the whole purpose of this project), I developed the algorithm from scratch. In the HelpFunctions.py file, you will find the functions used to perform K-means, mainly, Initialization of centroids, Expectation Step (assigning points to their respective clusters) and Maximization Step (computing the new centroids). We iterate through these two steps until we reach some number of Maximum Iterations. You can control most of these parameters.

There is also another interesting algorithm implemented within K-means, which is widely known as K-Means++. It is basically a "smart" probabilistic initialization procedure of centroids that improves the chances of converging to the global minimum of the objective function (distorsion or inertia). If you want to know how it works, head over here before reading my code : https://datasciencelab.wordpress.com/2014/01/15/improved-seeding-for-clustering-with-k-means/

	C) Visualizations

Aside from the main k-means scatterplot visualization, I have also added a graph of Inertia (or distorsion) as a function of the number of iterations (1 iteration = 1 E-Step + 1 M-Step). There is also a reference line showing the global minimum that the k-means could potentially reach (with an ideal initialization) under the current setting (choice of dataset, number of centroids k, initialization procedure). This was computed using Scikit-Learn's KMeans function as it would most likely converge to the global minimum with such easy 2D datasets. Basically this plot shows how some settings will not converge to optimal values of the objective function.

The second graph is a Silhouette Analysis graph, that helps assess the quality of the clustering, as in, how well each data point clusters (is it similar enough to its own cluster compared to other clusters). This plot aims to show that some datasets are just not clusterable, and some are not clusterable with K-means, but other algorithms (density clustering for example) could work better. 

	D) Code

I made sure to identify and name the different sections within my code. However, I feel that the code doesn't contain enough comments (it wasn't my objective). So if you can't understand something, as always, feel free to reach out.



**************
VII) LICENSE *
**************

Since this is a sizable project with a working prototype, I have decided to share it under the GNU GPLv2 license. Basically it's open source with very few restrictions.








