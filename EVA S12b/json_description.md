{
	"vests (1).jpg8283": {
		"filename": "vests (1).jpg",
		"size": 8283,
		"regions": [
			{
				"shape_attributes": {
					"name": "rect",
					"x": 120,
					"y": 79,
					"width": 95,
					"height": 101
				},
				"region_attributes": {
					"class_name": "vest"
				}
			}
		],
		"file_attributes": {
			"caption": "",
			"public_domain": "no",
			"image_url": ""
		}
	}
	

# Description
We have used the VGG image annotator tool to annotate four objects i.e. mask, hardhat, vest and boots. It has the option to
download the data in multiple format including json and csv formats. We are using json format. Structure of the json data 
has a dictionary with all the images inside it. Above we can see an example of one of the images. The key is the filename 
concatenated with file size. The dictionary value contains all the data regarding the bounding boxes.

The elements of dictionary are given as :
 1. "filename" - Name of the file in string format
 2. "size" - Size of the file as an integer in bytes
 3. "regions"- Contains detail regarding the bounding boxes - It is a list of dictionaries containing all the boxes in 
 the given image. Each element of the list is a dictionary with a structure described below:
    a). "shape_attributes" - Contains the detail regarding bounding boxes. It is a dictionary. 
	    The elements of the dictionary are as follows:
           1. "name"- It describes the shape of the box if it's a rectangle or circle or a polygon.
           2. "x" - It represents the x component of the centroid of bounding box - Horizontal position of the centroid.	
           3. "y" - It represents the y component of the centroid of bounding box - Vertical position of the centroid.
           4. "height" - Height of the bounding box
           5. "width"  - Width of the bounding box

    b). "region_attributes"	- Contains the detail regarding class name of a particular bounding box. It is a dictionary.

 4. "file_attributes" - Contains metadata regarding a particular image file. 
    a). "caption"
	b). "public_domain"
	c). "image_url"