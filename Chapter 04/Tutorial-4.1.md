#  Interactive Tutorial: 4.1 Build and Deploy Microservers in App Engine
 This tutorial will step you through the process required to deploy 3 microservices, one each in Node, Python and Go in a single GAE project and run them on a single GAE instance.
##  Step 1.1: Create the following directory structure:
```
gae-node-python-go — root directory for all 3 microservices
gae-node-python-go\node — root directory for Node microservice
gae-node-python-go\python — root directory for Python microservice
gae-node-python-go\go — root directory for Go microservice
 ```
Open Cloud Shell and on the CLI enter the following from your $Home directory.
```
mkdir gae-node-python-go
mkdir gae-node-python-go\node
mkdir gae-node-python-go\python
mkdir gae-node-python-go\go
```
#  Step 2. Building a Node Microservice
In this first step we are going to build and launch a very simple service written in Node.js and packaged as a microservice on App Engine. In this tutorial we will use Express to serve HTTP requests.
The files used in this tutorial can be found here:
```
git clone
```
Step 2.1: Open Cloud Shell and change to the Node.js the working directory:
```
cd  \gae-node-python-go
cd \node
```
Step 2.2: **Open Cloud Editor**. Browse to the  \gae-node-python-go\node directory and **click Create New File**. Name the file hello.js file and enter the code below and click Save.
```
```
The const port -  “process.env.PORT” variable sets the listening port for the webserver. 

Step 2.3: **Open the Terminal**: and enter:
```
 npm init
```
 and select the default values provided to create the package.json file.
Step 2.4  **Run install** and include Express
```
npm install --save express 
```
Step 2.5: We need to specify the start script, the Node version and the NPM version to GAE in the package.json file. Your final package.json file should look similar to below:
```
```
Step 2.6: Create the node-app.yaml file and enter the code below. 
```
```
Step 2.7: Open the Terminal and on the CLI enter the gcloud command to deploy Node microservice. This should take a few minutes.
```
gcloud app deploy node-app.yaml 
```
Step 2.8: When the build is successful completed enter:
```
gcloud app browse
```
 
You would be taken to https://[your-project-id].appspot.com/ in your default browser and you should see Node microservice in action.
# Step 3. Building and Deploying a Python Microservice
In this tutorial we will configure and deploy a Python microservice that uses Flask and Gunicorn to serve HTTP requests. 
Step 3.1: Open Terminal and change the directory:
```
cd  gae-node-python-go
cd \python
```
We will create, store and execute all commands for our Python microservice from this directory.
Step 3.2: **Next install Flask and gunicorn**
```
pip install flask
pip instal gunicorn
```
Step 3.3: **Open Shell Editor** and **create hello.py ** file and enter code below:
```
```
Step 3.4: Next **Create a new**python-app.yaml file using the code below:
```
```
Step 3.5: Now **create a text file** called requirements.txt file and **enter the code** below to list the latest versions we wish to install of our 2 dependencies, i.e. Flask and gunicorn.
```
```
Step 3.6: **Open The Terminal** and on the CLI deploy Python microservice. This should take a few minutes.
```
gcloud app deploy python-app.yaml 
```
Step 3.7: Run gcloud app browse -s python. You would be taken to https://python-dot-[your-project-id].appspot.com/ in your default browser and you should see Python microservice in action.

# Step 4.0: Building and Deploying Microservices on Go 
In this tutorial we will be using a Go microservice that uses the built-in net/http module to serve HTTP requests.
Step 4.1: Open the Terminal and change the working directory gae-node-python-go\go 
```
cd gae-node-python-go
cd \go
```
We will execute all commands for Go microservice from this directory.
 
Step 4.2: Open Editor and create the hello.go file using the code below:
```
```
Step 4.3: Next create a go-app.yaml file using the code below. 
 ```
```
Step 4.4: Run gcloud app deploy go-app.yaml command to deploy Go microservice. This should take a few minutes.
```
gcloud app deploy go-app.yaml
```
Step 4.5: Run gcloud app browse -s go. You would be taken to https://go-dot-[your-project-id].appspot.com/ in your default browser and you should see Go microservice in action.
