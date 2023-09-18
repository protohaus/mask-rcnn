raw_data_text = "Specify the folder to the image data files. \n \
This will be the basis for the leaf segmentation \n \
and subsequent training."

ai_weights_text = "Specify a folder to the weights file of \n \
the model. This will be a .h5 file and contains \n \
the latest training data. \n \
This will be used for the AI Segmentation part in \n \
the image segmentation and as the basis for the next \n \
training cycle in Training. \n \
The chosen image will be used to inspect the layers \n \
of the chosen weights file."

segmentation_text = "Mask RCNN needs information where each leaf \n \
is located in the image. \n \
This process is called image segmentation. \n \
If the raw data is already segmented, choose \n \
the appropriate json file. \n \
If the raw data is not yet segmented, there are \n \
two possibilities:\n \
Choose AI segmentation to let the AI segment the \n \
data. Inspect the result with sho results. \n \
If the results need tweaking hit manual segmentation. \n \
This will open up the VIA tool in your browser. \n \
In the browser window open up the folder and import \n \
the ai generated json. Now the polygons can be tweaked. \n \
Alternatively you can directly segment the data \n \
manually."

leaf_organizer_text = "In this section the best leafs from  the \n \
previous segmentation step are selected. \n \
Choose a folder to output the resulting leafs, \n \
then hit Start Leaf Organizer. \n \
This will take some time, because all images and masks\n \
are loaded. \n \
Each leaf from the dataset will be presented with \n \
the option to say yes or no. \n \
When the dataset is finished, hit Save Data to save the \n \
selected leafs."

leaf_collages_text = "Here, the single leafs from the previous \n \
step will be compiled into collages of overlapping \n \
leafs on different neutral backgrounds. \n \
First specify the background input directory and the \n \
collage output directory, then specify the number of \n \
samples. The folder need to be empty. \n \
A typical split of 80 to 20 is automatically generated. \n \
Specify the number of collages to generate."

training_text = "This is the training configuration page. \n \
Specify a folder for the logs. This is also the \n \
folder where the trained weights after training \n \
can be found.\n \
The training parameters are the layers that should be trained, \n \
the number of training epochs and the learning rate. \n \
Hitting the train button will freeze this window\n \
until the training is done. This will take several\n \
hours."

test_text = "This will validate the ai with the weights\n \
specified on the weights page with the validation set\n \
from the test folder. The result is displayed as a\n \
precisions-recall plot."