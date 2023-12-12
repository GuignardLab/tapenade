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
from scipy.optimize import linear_sum_assignment
import napari

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
        image_flip = tifffile.imread(rf'{folder_experiment}/{num_top}_top.tif')
        for ind_ch,ch in enumerate(channels) :
            tifffile.imwrite(folder_sample+rf"/raw/{name}_bot_{ch}.tif",image_bottom[:,ind_ch,:,:])#,imagej=True, metadata={'axes': 'TZYX'})
            tifffile.imwrite(folder_sample+rf"/raw/{name}_top_{ch}.tif",image_flip[:,ind_ch,:,:])#,imagej=True, metadata={'axes': 'TZYX'})



def manual_registration_fct(label_bottom, label_top) :
    """
    Finds the transformation between 2 sets of points in 3D.
    If the automatic registration can't find an accurate transformation :
    -open your bottom and top images on a visualization software (eg Napari) and create 2 label images of the same shape, one will contain the landmarks for the bottom image and the other for the top image.
    -add landmarks, preferentially sphere on the objects you can identify on both sides. Each object needs to have the same label on both sides, top and botttom. The more landmarks, the better.
    -then gives these 2 label arrays to the function manual_registration_fct, it will compute the transformation between the two sets of points

    Parameters
    ----------
    label_bottom : np.array
        label of the bottom image
    label_top : np.array
        label of the top image
    
    Returns
    -------
    translation and rotation to apply to the top image to register it on the bottom image
    In the following order : translation_x, translation_y, translation_z, rotation_x, rotation_y, rotation_z.
    """

    rg_bottom=regionprops(label_bottom)
    centroids_bottom=np.array([prop.centroid for prop in rg_bottom]).T
    rg_trop=regionprops(label_top)
    centroids_top=np.array([prop.centroid for prop in rg_trop]).T

    centermass_bottom = np.mean(centroids_bottom,axis=1).reshape(3,1)
    centermass_top = np.mean(centroids_top,axis=1).reshape(3,1)

    centered_coords_bottom=centroids_bottom-centermass_bottom
    centered_coords_top=centroids_top-centermass_top


    C = centered_coords_bottom @ centered_coords_top.T
    U,S,Vt = np.linalg.svd(C)

    Rot = U @ Vt

    translation = centermass_bottom-centermass_top
    # Convert the rotation matrix to a Rotation object
    rotation = Rotation.from_matrix(Rot)
    # quaternion = (rotation.as_quat())
    # rot_x = quaternion

    rotation_angles = rotation.as_euler('zyx', degrees=True)
    # Extract individual rotation angles
    rotation_z, rotation_y, rotation_x = rotation_angles

    return(translation[2],translation[1],translation[0],int(rotation_x),int(rotation_y),int(rotation_z))

def register(path_data:str, path_to_bin:str, sample_id:str, channel:str, input_voxel:tuple=[1,1,1],
            output_voxel:tuple=[1,1,1],compute_trsf:int=1, init_trsfs=[["flip", "Y", "flip", "Z", "trans", "Z", -10,]],
            trsf_type:str='rigid', depth:int=3, save_json:bool=False) :
    
    # Register the two sides of the sample, using the previously computed transformation (if any) or computing a new one

    
    # Parameters
    # ----------
    # path_data : str
    #     main path to the data folder (the registration will be saved in a subfolder)
    # path_to_bin : str
    #     path to the bin folder : necessary to save the registered data : should be something like 'C:\Users\user\Anaconda3\envs\my_environment\Library\bin'

    # sample_id : str
    # channel : str
    #     channel to register : only one here
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
    #     
    # save_json : bool, optional
    #     if True, saves the parameters in a json file, in the main folder. By default False.
    




    data = {
            "path_to_bin": path_to_bin, #necessary to register the data
            "path_to_data": rf"{path_data}/{sample_id}/raw/",
            "ref_im": rf"{sample_id}_bot_{channel}.tif",
            "flo_ims": [rf"{sample_id}_top_{channel}.tif"
            ],
            "compute_trsf": compute_trsf
            ,
            "init_trsfs":init_trsfs,
            "trsf_paths": [rf"{path_data}/{sample_id}/trsf/"],
            "trsf_types": [trsf_type],
            "ref_voxel": input_voxel,
            "flo_voxels": [ input_voxel],
            "out_voxel":output_voxel,
            "test_init": 0,
            "apply_trsf": 1,
            "out_pattern": rf"{path_data}/{sample_id}/registered",
            "begin" : 1,
            "end":1,
            "bbox_out": 1,
            "registration_depth":depth,
        }

    if save_json :
        json_string=json.dumps(data)
        with open(path_data+'\parameters.json','w') as outfile :
            outfile.write(json_string)

    tr = registrationtools.SpatialRegistration(data)
    tr.run_trsf()

def check_napari(folder_experiment:str,
                 sample_id:str,
                 channel,
                 scale:tuple=(1,1,1)) :
    """
    Opens the registered images in napari , to check how they overlap

    Parameters
    ----------
    folder_experiment : str
        path to the main folder
    sample_id : str
    channel : str
        name of the channel to look at
    scale : tuple, optional
        scale of the image (should be the same ), by default (1,1,1)
    """


    viewer=napari.Viewer()
    reg_flip1 = tifffile.imread(rf'{folder_experiment}/{sample_id}/registered/{sample_id}_bot_{channel}.tif') 
    reg_flip2 = tifffile.imread(rf'{folder_experiment}/{sample_id}/registered/{sample_id}_top_{channel}.tif')
    viewer.add_image(reg_flip1,colormap='cyan',name=rf'{sample_id}_bot_{channel}',scale=scale)
    viewer.add_image(reg_flip2,colormap='red',opacity=0.4,name=rf'{sample_id}_top_{channel}',scale=scale)
    napari.run()

def fuse_sides (folder_experiment:str,
                sample_id,
                channels:list,
                mode:str='linear',
                folder_output:str='',
                axis:int=0):
    """
    Fuse the two sides of the sample, using the previously registered images

    Parameters
    ----------
    folder_experiment : str
        path to the main folder
    channels : list
        list of the names of the channels
    sample_id : str
    mode : str, optional
        mode of fusion : linear or ?, by default 'linear'
    folder_output : str, optional
        path to the output folder, if None, will be the same as the experiment folder
    axis : int, optional
        axis along which the fusion is done, by default 0 (z axis)
    """


    if folder_output=='' : #if user does not give a output path, then its saved in 'output' folder inside the 'sample' folder
            os.mkdir(os.path.join(rf'{folder_experiment}\{sample_id}',"fused"))
            folder_output=rf'{folder_experiment}\{sample_id}\fused'
        
    for ch in channels :
    ###Takes the previously saved images (two registered sides) to merge them
        reg_flip1 = tifffile.imread(rf'{folder_experiment}/{sample_id}/registered/{sample_id}_bot_{ch}.tif')
        reg_flip2 = tifffile.imread(rf'{folder_experiment}/{sample_id}/registered/{sample_id}_top_{ch}.tif')
        #mask of the 2 sides to weight the image by the distance to the edges (intensity normalization)
        mask_flip1 = reg_flip1>0
        mask_flip2 = reg_flip2>0
        mask_fused = mask_flip1 & mask_flip2
        if mode =='linear' :
            cumsum = np.cumsum(mask_fused,axis=axis).astype(np.float16)
            cumsum_normalized = cumsum/np.max(cumsum)
            w2 = cumsum_normalized #IN CASE OF PB TRY THIS FIRST : if positive sign to translation : w2=
            w1=1-w2
            #saves the result in the fused folder
            fusion = (reg_flip1*w1+ reg_flip2*w2).astype(np.uint16)

        tifffile.imwrite(folder_output+rf'/{sample_id}_{ch}_fused.tif', fusion)

def write_hyperstacks(folder_experiment:str,
                      sample_id:str,
                      channels:list) :
    image = tifffile.imread(rf'{folder_experiment}\{sample_id}\fused\{sample_id}_{channels[0]}_fused.tif',dtype=np.int16)
    (z,x,y)=image.shape
    new_image = np.zeros((z,len(channels),x,y))
    print(new_image.shape)
    for ch in range(len(channels)) :
        one_channel = tifffile.imread(rf'{folder_experiment}\{sample_id}\fused\{sample_id}_{channels[ch]}_fused.tif',dtype=np.int16)
        new_image[:,ch,:,:]=one_channel
        print(one_channel.shape,new_image.shape)
    tifffile.imwrite(rf'{folder_experiment}\{sample_id}_registered.tif',new_image)






def reconstruct_foo():
    print("reconstruct")
    return -1


def pipeline_reconstruction(*args):  # image1, image2, sigma, ...
    print(*args)


def script_run():

    # Parse the arguments
    pipeline_reconstruction(1, 2, 3, 4, 5)
