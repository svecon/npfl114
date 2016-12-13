# HW-11 nli (3-15pts, due Jan 09)

Try solving the Native Language Identification task with highest accuracy possible, ideally beating current state-of-the-art.

The dataset is available under a restrictive license, so the details about how to obtain it have been sent by email to the course participants. If you have not received it, please write me an email and I will send you the instructions directly.

Your goal is to achieve highest accuracy on the test data. The dataset you have does not contain test annotations, so you cannot measure test accuracy directly. Instead, you should measure development accuracy and finally submit test annotations for the model with best development accuracy.

You can load the dataset using the labs09/nli_dataset.py module. You can start with the labs09/nli-skeleton.py file, which uses the labs09/nli_dataset.py module to load the data, passes the data to the network and finally produces test annotations using the model achieving highest development accuracy.

In order to solve the task, send me the test set annotations and also the source code. I will evaluate the test set annotations using the labs09/nli_evaluate.py script. Every working solution will get 3 points, and you will get additional points accordint to your test set accuracy – the best solution will get a total of 15 points, the next one 14, and so on. Also everyone beating state-of-the-art will get a total of 15 points.
