# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Information to the project ##
The task was to build an image classifier with pytorch to classify the oxford flowers dataset. Therefore i used the densenet121 net and changed the classifying layer. The second task was to write a command line application for training and one for predicting new images.

## Explanation ##
There are the following files to use:
<ul>
  <li>Image Classifier Project.ipynb: This is the Jupyter Notebook i developed the code and trained the classifier. For submission a html-file is included. </li>
  <li>train.py: The command line application for training the classifier. You can choose between three architectures and if GPU is available using the GPU.</li>
  <li>predict.py: The command line application for predicting new pictures. It is possible to choose the top k predictions for a picture.</li>
  <li>cat_to_name.json: This file includes the class names for the flowers and the corresponding class numbers. </li>
</ul>

You need the flowers dataset as follows:
<ol>
  <li>/flowers/train/xx/yy.jpg</li>
  <li>/flowers/test/xx/yy.jpg</li>
  <li>/flowers/valid/xx/yy.jpg</li>
</ol>
in the same folder as the *.py files with xx as the categorical number of the flowers and yy as the filenames of the flowers.

## Acknowledgements ##
This project was my final project at the Udacity Nanodegree program "AI Programming with Python". I am super thankful for this experience and proud to be certified with this nanodegree program.

## License ##
This project is open source and available under the [Udacity License]
