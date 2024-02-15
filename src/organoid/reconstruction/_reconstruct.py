import numpy as np
import math
import matplotlib.pyplot as plt
from xml.dom import minidom
from scipy.optimize import linear_sum_assignment
import numpy as np
import math
import matplotlib.pyplot as plt
import json
import os
from skimage.measure import regionprops
from scipy.spatial.transform import Rotation
import registrationtools
import tifffile
from xml.dom import minidom
import napari

#Remains to be done/solved :
#should we keep the names 'bottom' and 'top' ? or should we use 'reference' and 'floating' ? view1 and view 2 ?

def extract_positions(path_positions:str):
    """
    Extract the positions of the objects from the xml file (saved during the acquisition)
    Parameters
    ----------
    path_positions : str
        path to the xml file
    """
    positions = minidom.parse(path_positions)

    pos_x = positions.getElementsByTagName('dXPosition')
    pos_y = positions.getElementsByTagName('dYPosition')

    xpos=np.zeros(pos_x.length)
    ypos=np.zeros(pos_y.length)

    for i in range(pos_y.length):
        xpos[i]=pos_x[i].attributes['value'].value
        ypos[i]=pos_y[i].attributes['value'].value

    return(xpos,ypos)

def plot_positions(path_bottom_positions:str,path_top_positions:str) :
    """
    Plot the positions of the objects from the xml file, to visualize and manually check what number have to be associated
    Parameters
    ----------
    path_bottom_positions : str
        path to the xml file for the bottom view
    path_top_positions : str
        path to the xml file for the top view

    """        
    (xpos_b,ypos_b)=extract_positions(path_bottom_positions)
    (xpos_t,ypos_t)=extract_positions(path_top_positions)   

    list_number_b= [i+1 for i in range(len(xpos_b))] #the objects will have an index going from 1 to the total number, instead of having the id chosen during the aquisition. Problem ?
    fig, ax = plt.subplots()
    ax.scatter(xpos_b, ypos_b,label='bottom')

    for i, txt in enumerate(list_number_b):
        ax.annotate(txt, (xpos_b[i], ypos_b[i]))

    list_number_t= [i+1 for i in range(len(xpos_t))]
    ax.scatter(xpos_t, -ypos_t,label='top')

    for i, txt in enumerate(list_number_t):
        ax.annotate(txt, (xpos_t[i],-ypos_t[i]))

    plt.legend()


def associate_top_bottom(path_bottom_positions:str,path_top_positions:str):
    """
    Associate the objects from the bottom view with the objects from the top view, by solving a linear sum assignement between the two distribution
    Parameters
    ----------
    path_bottom_positions : str
        path to the xml file for the bottom view
    path_top_positions : str
        path to the xml file for the top view
    
    Returns
    -------
    2 lists of indices, bottom then top, sorted according to the asssgnement

    """

    (xpos_b,ypos_b)=extract_positions(path_bottom_positions)
    (xpos_t,ypos_t)=extract_positions(path_top_positions)  

    cost=np.zeros((len(xpos_b),len(xpos_t)))
    for i in range(len(xpos_b)):
        for j in range(len(xpos_t)):
            cost[i,j]=math.sqrt(math.pow(xpos_b[i]-xpos_t[j],2)+math.pow(ypos_b[i]-ypos_t[j],2))
    row_ind, col_ind = linear_sum_assignment(cost)

    return(list(row_ind+1),list(col_ind+1))


    # assert xpos_b.shape==xpos_t.shape==ypos_b.shape==ypos_t.shape



def create_folders(folder_experiment:str,
                   list_bottom:list,
                   list_top:list,
                   channels:list,
                   folder_output:str=''):
    """
    Creates the folders to save the registered images, and save the channels separately

    Parameters
    ----------
    folder_experiment : str
        path to the main folder
    list_bottom : list
        list of the numbers of the bottom images
    list_top : list
        list of the numbers of the top images
    channels : list
        list of the names of the channels
    folder_output : str, optional
        path to the output folder, if None, will be the same as the experiment folder
    """
        
    if folder_output=='' :
        folder_output=folder_experiment

    list_samples=[str(i+1) for i in range(len(list_bottom))] #to look at every sample from the acquisition

    for ind_g,name in enumerate(list_samples) : #for each sample, creates a dedicated folder and save the channels separately
        folder_sample = rf'{folder_experiment}/{name}/'

    #creates paths for the output files
        os.mkdir(os.path.join(folder_experiment,name))
        os.mkdir(os.path.join(folder_sample,"trsf"))
        os.mkdir(os.path.join(folder_sample,"raw"))
        os.mkdir(os.path.join(folder_sample,"registered"))
        num_bottom = list_bottom[ind_g]
        num_top = list_top[ind_g]
        image_bottom = tifffile.imread(rf'{folder_experiment}/{num_bottom}_bottom.tif')
        image_top = tifffile.imread(rf'{folder_experiment}/{num_top}_top.tif')
        for ind_ch,ch in enumerate(channels) :
            tifffile.imwrite(folder_sample+rf"/raw/{name}_bot_{ch}.tif",image_bottom[:,ind_ch,:,:])#,imagej=True, metadata={'axes': 'TZYX'})
            tifffile.imwrite(folder_sample+rf"/raw/{name}_top_{ch}.tif",image_top[:,ind_ch,:,:])#,imagej=True, metadata={'axes': 'TZYX'})



def manual_registration_fct(reference_landmarks, floating_landmarks):
#stolen from https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
    """
    Finds the transformation between 2 sets of points in 3D.
    If the automatic registration can't find an accurate transformation :
    -open your bottom and top images on a visualization software (eg Napari) and create 2 label images of the same shape, one will contain the landmarks for the bottom image and the other for the top image.
    -add landmarks, preferentially sphere on the objects you can identify on both sides. Each object needs to have the same label on both sides, top and botttom. The more landmarks, the better.
    -then gives these 2 label arrays to the function manual_registration_fct, it will compute the transformation between the two sets of points

    Parameters
    ----------
    reference_landmarks : np.array
        This image will be the reference, the 'fixed' one
    floating_landmarks : np.array
        This image will be the floating image, the one that will be registered onto the reference image
    
    Returns
    -------
    translation and rotation to apply to the top image to register it on the bottom image
    In the following order : rotation_z, rotation_y, rotation_x, translation_z, translation_y, translation_x.
    """

    rg_ref=regionprops(reference_landmarks)
    centroids_ref=np.array([prop.centroid for prop in rg_ref]).T
    rg_float=regionprops(floating_landmarks)
    centroids_float=np.array([prop.centroid for prop in rg_float]).T


    assert centroids_ref.shape == centroids_float.shape

    num_rows, num_cols = centroids_ref.shape
    if num_rows != 3:
        raise Exception(f"matrix centroids_ref is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = centroids_float.shape
    if num_rows != 3:
        raise Exception(f"matrix centroids_float is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centermass_ref = np.mean(centroids_ref, axis=1)
    centermass_float = np.mean(centroids_float, axis=1)

    # ensure centermasss are 3x1
    centermass_ref = centermass_ref.reshape(-1, 1)
    centermass_float = centermass_float.reshape(-1, 1)

    # subtract mean
    centroids_ref_centered = centroids_ref - centermass_ref
    centroids_float_centered = centroids_float - centermass_float

    H = centroids_ref_centered @ np.transpose(centroids_float_centered)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = R @ centermass_ref + centermass_float
    rotation = Rotation.from_matrix(R)
    rotation_angles = rotation.as_euler('zyx', degrees=True)
    rotation_z, rotation_y, rotation_x = rotation_angles
    trans_z,trans_y,trans_x=t


    return (rotation_z,rotation_y,rotation_x,trans_z,trans_y,trans_x)


def register(path_data:str,
             path_transformation:str,
             path_registered_data:str,
             path_to_bin:str,
             reference_image:str,
             floating_image:str,
             input_voxel:tuple=[1,1,1],
             output_voxel:tuple=[1,1,1],
             compute_trsf:int=1,
             init_trsfs=[["flip", "Y", "flip", "Z", "trans", "Z", -10,]],
             test_init:int=0,
             trsf_type:str='rigid',
             depth:int=3,
             save_json:str='') :


##this is under comments because if i put quotation marks, the importation fails (?)
    # Register the two sides of the sample, using the previously computed transformation (if any) or computing a new one


    # Parameters
    # ----------
    # path_data : str
    #     path to the raw images
    # path_transformation : str
    #     path to the folder where the transformations files are saved
    # path_registered_data : str
    #     path where the registered images will be saved
    # path_to_bin : str
    #     path to the bin folder : necessary to save the registered data : should be something like 'C:\Users\user\Anaconda3\envs\my_environment\Library\bin'
    # reference_image : str
    #     name of the reference image, the 'fixed' one
    # floating_image : str
    #     name of the floating image, the one that will be registered onto the reference image
    # input_voxel : tuple, optional
    #     voxel size of the input image, by default [1,1,1]
    # output_voxel : tuple, optional
    #     voxel size of the output image, by default [1,1,1]. Can be different than the input voxel size.
    # compute_trsf : int, optional
    #     1 if the transformation has to be computed, 0 if it already exists. If you have multiple channels of the same image, it is recommended to pick one expressed homogeneously as teh reference, register this channel using compute_trsf=1.
    #     Then you can use compute_trsf=0, the algo will find the pre-existing transformation to register the other channels onto the reference channel.
    # init_trsfs : list, optional.
    #     You can use flipping (flip), rotations (rot), translation (trans). Specify the axis X,Y or Z after the sample_id of the transformation. For rotations and translation, precise the value (angle or distance) after the axis.
    #     Needs 2 sets of brackets.
    #     Example : [["flip", "Z", "trans", "Z", -10,"trans","Y",100,"rot","X",-29,"rot","Y",41,"rot","Z",-2]]
    # trsf_type : str, optional
    #     type of transformation to compute : rigid, affine. By default rigid.
    # depth : int, optional
        
    # save_json : bool, optional
    #     if True, saves the parameters in a json file, in the main folder. By default False.


    data = {
            "path_to_bin": path_to_bin, #necessary to register the data
            "path_to_data": path_data,
            "ref_im": reference_image, #rf"{sample_id}_bot_{channel}.tif",
            "flo_ims": [floating_image], # [rf"{sample_id}_top_{channel}.tif"],
            "compute_trsf": compute_trsf,
            "init_trsfs":init_trsfs,
            "trsf_paths": [path_transformation],
            "trsf_types": [trsf_type],
            "ref_voxel": input_voxel,
            "flo_voxels": [ input_voxel],
            "out_voxel":output_voxel,
            "test_init": test_init,
            "apply_trsf": 1,
            "out_pattern": path_registered_data,
            "begin" : 1,
            "end":1,
            "bbox_out": 1,
            "registration_depth":depth,
        }

    if save_json is not '':
        json_string=json.dumps(data)
        with open(save_json+'\parameters.json','w') as outfile :
            outfile.write(json_string)

    tr = registrationtools.SpatialRegistration(data)
    tr.run_trsf()

def check_napari(path_registered_data:str,
                 reference_image:str,
                 floating_image:str,
                 scale:tuple=(1,1,1)) :
    """
    Opens the registered images in napari , to check how they overlap

    Parameters
    ----------
    folder_experiment : str
        path to the main folder
    sample_id : str
    scale : tuple, optional
        scale of the image (should be the same ), by default (1,1,1)
    """


    viewer=napari.Viewer()
    ref_im = tifffile.imread(rf'{path_registered_data}/{reference_image}') 
    float_im = tifffile.imread(rf'{path_registered_data}/{floating_image}')
    viewer.add_image(ref_im,colormap='cyan',name=rf'{reference_image}',scale=scale)
    viewer.add_image(float_im,colormap='red',blending='additive',name=rf'{floating_image}',scale=scale)
    napari.run()


def sigmoid(x,x0,p):
    """
    Sigmoid function, to weight the images by the distance to the edges
    z0 gives the middle of the sigmoid, at which weight=0.5
    p gives the slope. 5 corresponds to a low slope, wide fusion width and 25 to a strong slope, very thin fusion width.
    """
    # for z in range(len(z)) :
    return (1 / (1 + np.exp(-p*(x-x0))))
    
def fuse_sides (path_registered_data:str,
                reference_image_reg:str,
                floating_image_reg:str,
                folder_output:str='',
                name_output:str='fusion',
                slope_coeff:int=20,
                axis:int=0):
    """
    Fuse the two sides of the sample, using the previously registered images

    Parameters
    ----------
    path_registered_data : str
        path to the registered images
    reference_image_reg : str
        name of the reference image, the 'fixed' one
    floating_image_reg : str
        name of the floating image, the one that will be registered onto the reference image
    folder_output : str
        path to the folder where the fused image will be saved
    name_output : str, optional
        name of the output which is the fused image , by default 'fusion'
    axis : int, optional
        axis along which the fusion is done, by default 0 (z axis)
    """
        
    ###Takes the previously saved images (two registered sides)
    ref_image = tifffile.imread(rf'{path_registered_data}/{reference_image_reg}')
    float_image = tifffile.imread(rf'{path_registered_data}/{floating_image_reg}')
    mask_r = ref_image>0
    mask_f = float_image>0
    mask_fused = mask_r & mask_f
    cumsum = np.cumsum(mask_fused,axis=axis).astype(np.float16) #we take the cumulative sum  along the fusion axis, this will give me a linear weight.
    cumsum_normalized = cumsum/np.max(cumsum)

    #apply a sigmoid function to the linear weights, to get a smooth transition between the two sides
    w2=sigmoid(cumsum_normalized,x0=0.5,p=slope_coeff)
    w1=1-w2

    fusion = (ref_image*w1+ float_image*w2).astype(np.uint16)

    tifffile.imwrite(rf'{folder_output}/{name_output}', fusion)


def write_hyperstacks(path:str,
                      sample_id:str,
                      channels:list) :
    image = tifffile.imread(rf'{path}\{sample_id}_{channels[0]}_fused.tif',dtype=np.int16)
    (z,x,y)=image.shape
    new_image = np.zeros((z,len(channels),x,y))
    print(new_image.shape)
    for ch in range(len(channels)) :
        one_channel = tifffile.imread(rf'{path}\{sample_id}_{channels[ch]}_fused.tif',dtype=np.int16)
        new_image[:,ch,:,:]=one_channel
        print(one_channel.shape,new_image.shape)
    tifffile.imwrite(rf'{path}\{sample_id}_registered.tif',new_image)






def reconstruct_foo():
    print("reconstruct")
    return -1


def pipeline_reconstruction(*args):  # image1, image2, sigma, ...
    print(*args)


def script_run():

    # Parse the arguments
    pipeline_reconstruction(1, 2, 3, 4, 5)
