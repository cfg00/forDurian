#conda activate cellpose&&python D:\Carlos\Scripts\workerDECONVOLUTIOND106.py
#
######### Please add documentation
### This is experiemnt D106, we are analyzing the big tad, with only 200 FOV. Date started: 9/13/23 by Carlos

#Have you copmputed the PSF?
#################################################################
import keyboard
from multiprocessing import Pool, TimeoutError
import time,sys
import os,sys,numpy as np

master_analysis_folder = r'/home/cfg001/Desktop/D104-main'
sys.path.append(master_analysis_folder)
from ioMicro import *
#standard is 4, its number of colors +1 
ncols = 4
#\\merfish7\merfish7v2\20230515D101Myh67d15
# r'\\merfish7\merfish7v2\20230515D101Myh67d15\AnalysisDeconvolveBBSingleColor'
#save_folder =r'\\merfish7\merfish7v2\20230515D101Myh67d15\AnalysisDeconvolveCG'
psf_file = r'/home/cfg001/Desktop/D104-main/psfs/psf_647_Kiwi.npy'
#\\merfish9\volume1\20230908_D106Luo\DNA\AnalysisDeconvolveCG
save_folder = r'/mnt/merfish8v1/Carlos_Analysis/AnalysisDeconvolveCG'

flat_field_fl = r'/home/cfg001/Desktop/D104-main/flat_field/Kiwi__med_col_raw'

def compute_drift(save_folder,fov,all_flds,set_,redo=False):
    """
    save_folder where to save analyzed data
    fov - i.e. Conv_zscan_005.zarr
    all_flds - folders that contain eithger the MERFISH bits or control bits or smFISH
    set_ - an extra tag typically at the end of the folder to separate out different folders
    """
    #print(len(all_flds))
    #print(all_flds)
    gpu=True
    # defulat name of the drift file 
    drift_fl = save_folder+os.sep+'driftNew_'+fov.split('.')[0]+'--'+set_+'.pkl'
    
    iiref = None
    fl_ref = None
    previous_drift = {}
    if not os.path.exists(drift_fl) or redo:
        redo = True
    else:
        try:
            drifts_,all_flds_,fov_,fl_ref = pickle.load(open(drift_fl,'rb'))
            all_tags_ = np.array([os.path.basename(fld)for fld in all_flds_])
            all_tags = np.array([os.path.basename(fld)for fld in all_flds])
            iiref = np.argmin([np.sum(np.abs(drift[0]))for drift in drifts_])
            previous_drift = {tag:drift for drift,tag in zip(drifts_,all_tags_)}

            if not (len(all_tags_)==len(all_tags)):
                redo = True
            else:
                if not np.all(np.sort(all_tags_)==np.sort(all_tags)):
                    redo = True
        except:
            os.remove(drift_fl)
            redo=True
    if redo:
        fls = [fld+os.sep+fov for fld in all_flds]
        if fl_ref is None:
            fl_ref = fls[len(fls)//2]
        obj = None
        newdrifts = []
        all_fldsT = []
        for fl in tqdm(fls):
            fld = os.path.dirname(fl)
            tag = os.path.basename(fld)
            new_drift_info = previous_drift.get(tag,None)
            if new_drift_info is None:
                if obj is None:
                    obj = fine_drift(fl_ref,fl,sz_block=600)
                else:
                    obj.get_drift(fl_ref,fl)
                new_drift = -(obj.drft_minus+obj.drft_plus)/2
                new_drift_info = [new_drift,obj.drft_minus,obj.drft_plus,obj.drift,obj.pair_minus,obj.pair_plus]
            newdrifts.append(new_drift_info)
            all_fldsT.append(fld)
            pickle.dump([newdrifts,all_fldsT,fov,fl_ref],open(drift_fl,'wb'))
def compute_drift_features(save_folder,fov,all_flds,set_,redo=False,gpu=True):
    fls = [fld+os.sep+fov for fld in all_flds]
    for fl in fls:
        get_dapi_features(fl,save_folder,set_,gpu=gpu,im_med_fl = flat_field_fl+r'3.npz',
                    psf_fl = psf_file)
def main_do_compute_fits(save_folder,fld,fov,icol,save_fl,psf,old_method,icol_flat):
    '''
    Inputs: save_folder -> this is on the NAS,where the output is saved
    
    '''
    im_ = read_im(fld+os.sep+fov)
    im__ = np.array(im_[icol],dtype=np.float32)
    
    if old_method:
        ### previous method
        im_n = norm_slice(im__,s=30)
        #Xh = get_local_max(im_n,500,im_raw=im__,dic_psf=None,delta=1,delta_fit=3,dbscan=True,
        #      return_centers=False,mins=None,sigmaZ=1,sigmaXY=1.5)
        Xh = get_local_maxfast_tensor(im_n,th_fit=500,im_raw=im__,dic_psf=None,delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5,gpu=False)
    else:
        
        fl_med = flat_field_fl+str(icol)+'.npy'

        
        if os.path.exists(fl_med):
            im_med = np.array(np.load(fl_med)['im'],dtype=np.float32)
            im_med = cv2.blur(im_med,(20,20))
            im__ = im__/im_med*np.median(im_med)
        try:
            Xh = get_local_max_tile(im__,th=3600,s_ = 500,pad=100,psf=psf,plt_val=None,snorm=30,gpu=True,
                                    deconv={'method':'wiener','beta':0.0001},
                                    delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5)
        except:
            Xh = get_local_max_tile(im__,th=3600,s_ = 500,pad=100,psf=psf,plt_val=None,snorm=30,gpu=False,
                                    deconv={'method':'wiener','beta':0.0001},
                                    delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5)
    np.savez_compressed(save_fl,Xh=Xh)
def compute_fits(save_folder,fov,all_flds,redo=False,ncols=ncols,
                psf_file = psf_file,try_mode=True,old_method=False,redefine_color=None):
    
    psf = np.load(psf_file)
    for ifld,fld in enumerate(tqdm(all_flds)):
        if redefine_color is not None:
            ncols = len(redefine_color[ifld])
        for icol in range(ncols-1):
            ### new method
            if redefine_color is None:
                icol_flat = icol
            else:
                #print("ifld is: "+str(ifld))
                #print("icol is: "+str(icol))
                icol_flat = redefine_color[ifld][icol]
            tag = os.path.basename(fld)
            save_fl = save_folder+os.sep+fov.split('.')[0]+'--'+tag+'--col'+str(icol)+'__Xhfits.npz'
            if not os.path.exists(save_fl) or redo:
                if try_mode:
                    try:
                        main_do_compute_fits(save_folder,fld,fov,icol,save_fl,psf,old_method,redefine_color)
                    except:
                        print("Failed",fld,fov,icol)
                else:
                    main_do_compute_fits(save_folder,fld,fov,icol,save_fl,psf,old_method,redefine_color)
                    
def compute_decoding(save_folder,fov,set_):
    dec = decoder_simple(save_folder,fov,set_)
    complete = dec.check_is_complete()
    if complete==0:
        dec.get_XH(fov,set_,ncols=ncols)#number of colors match
        dec.XH = dec.XH[dec.XH[:,-4]>0.25] ### keep the spots that are correlated with the expected PSF for 60X
        dec.load_library(lib_fl = r'\\192.168.0.10\bbfishdc13\codebook_0_New_DCBB-300_MERFISH_encoding_2_21_2023.csv',nblanks=-1)
        dec.get_inters(dinstance_th=2,enforce_color=True)# enforce_color=False
        dec.get_icodes(nmin_bits=4,method = 'top4',norm_brightness=-1)
        
def main_f(set_ifov,try_mode=True):
    '''
    CHANGE PATH HERE
    '''
    #20230122_R120PVTS32RDNA for the data used in grant
    #r'\\merfish6\merfish6v2\DNA_FISH\20230212_R120PVTS32RDNABDNF\Analysis'
     ### save folder
    if not os.path.exists(save_folder): os.makedirs(save_folder)
    set__ = ''
   # r'\\merfish6\merfish6v2\DNA_FISH\20230212_R120PVTS32RDNABDNF\'
    
   
    #V:\20230515D101Myh67d15
    all_flds = []
    redefine_color = []
   
    
    
    
    all_flds_ = glob.glob(r'/mnt/merfish11/20230908_D106Luo/DNA/H*B*'+set__) ################################# modify for 3-col small tad 
    all_flds_ += glob.glob(r'/mnt/merfish9/20230908_D106Luo/DNA/H*B*'+set__)
    all_flds_+= glob.glob(r'/mnt/merfish10/20231031_D106Luo/DNA/H*R*'+set__)
    #all_flds_ = np.random.permutation(all_flds_)

    redefine_color += [[0,1,2,3]]*len(all_flds_) #### three signal color
    all_flds+=all_flds_
    
    #same rationale as cc1, not complete so not bothering
    #all_flds = [s for s in all_flds if "H103" not in s]
    
    set_,ifov = set_ifov
    all_flds = [fld.replace(set__,set_) for fld in all_flds]
    fovs_fl = save_folder+os.sep+'fovs__'+set_+'.npy'
    if not os.path.exists(fovs_fl):
        fls = glob.glob(all_flds[0]+os.sep+'*.zarr')
        fovs = [os.path.basename(fl) for fl in fls]
        np.save(fovs_fl,fovs)
    else: #######################################################################################################
        fovs = np.load(fovs_fl)
       # print("The amount of FOVS being processed: " + str(len(fovs))+ "please confirm!")
       # val = input("lol ")
       # print("Thanks! Starting now...")
        
    if ifov<len(fovs):
        fov = fovs[ifov]
        
        print("Computing fitting on: "+str(fov))
        print(len(all_flds),all_flds)
        compute_fits(save_folder,fov,all_flds,redo=False,try_mode=try_mode,redefine_color=redefine_color)
        print("Computing drift on: "+str(fov))
        try:
            print("try mode on FOV: " + str(fov))
            compute_drift_features(save_folder,fov,all_flds,set_,redo=False,gpu=True)
        #compute_drift(save_folder,fov,all_flds,set_,redo=False)
        except:
            print("*******************issues with FOV: " + str(fov))
        #compute_decoding(save_folder,fov,set_)
        
    return set_ifov
if __name__ == '__main__':
    # start 4 worker processes
    items = [(set_,ifov)for set_ in ['']
                        for ifov in range(10000)]
    items.reverse()
    print(items)
    print(save_folder)
    if True:
        with Pool(processes=4) as pool:
            print('starting pool')
            
            #try:
                #for i in range(0,201):
            print("not engaging with try mode...")
            result = pool.map(main_f,items)
            #except:
            #    print("failed at: " )

            
