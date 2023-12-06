from organoid.segmentation import predict_stardist
import numpy as np

def test_pred():
    array=np.ones((50,512,512))
    pred = predict_stardist(array,
                            model_path=rf'C:\Users\gros\Desktop\CODES\Alice_Segmentation\Stardist\models\lennedist_3d_grid222_rays64',
                            input_voxelsize=(0.3,0.2,0.2),normalize_input=True)
    
    assert pred.shape==array.shape
