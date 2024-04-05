from fundamentals.matlab_class import Struct
from parameters.basic_optical_parameters import basic_optical_parameters
import numpy as np

def update_struct(base_struct, update):
    updated_struct = base_struct

    for ii, field in enumerate(update.fields()[ii]):
        if field in updated_struct.fields():
            updated_struct.field = update.field
        else:
            raise KeyError('Unexpected field : {} was set ?! Verify the field name of your parameter structures.'.format(field))
        
    return updated_struct

def circshift(array, shift_amounts_list):
    """Performs a circular shift along multiple axes."""
    for axis, shift in enumerate(shift_amounts_list):
        array = np.roll(array, shift, axis=axis)
    return array

def derive_optical_tool(optical_parameter=None, use_gpu=False):
    params = basic_optical_parameters()

    if optical_parameter is not None:
        params = update_struct(params, optical_parameter)

    utility = Struct()

    # image space
    utility.image_space=Struct()
    utility.image_space.res=np.empty((3,3))
    utility.image_space.res[0]=params.resolution[0]
    utility.image_space.res[1]=params.resolution[1]
    utility.image_space.res[2]=params.resolution[2]
    utility.image_space.size=np.empty((3,3))
    utility.image_space.size[0]=params.size[0]
    utility.image_space.size[1]=params.size[1]
    utility.image_space.size[2]=params.size[2]
    utility.image_space.coor=np.empty((3,3))
    utility.image_space.coor[0]=np.array([i-int(params.size[0]/2) for i in range(params.size[0])])
    utility.image_space.coor[1]=np.array([i-int(params.size[1]/2) for i in range(params.size[1])])
    utility.image_space.coor[2]=np.array([i-int(params.size[2]/2) for i in range(params.size[2])])
    utility.image_space.coor[0]=utility.image_space.coor[0]*params.resolution[0]
    utility.image_space.coor[1]=utility.image_space.coor[1]*params.resolution[1]
    utility.image_space.coor[2]=utility.image_space.coor[2]*params.resolution[2]
    utility.image_space.coor[0]=np.reshape(utility.image_space.coor[0],(params.size[0],1,1))
    utility.image_space.coor[1]=np.reshape(utility.image_space.coor[1],(1,params.size[1],1))
    utility.image_space.coor[2]=np.reshape(utility.image_space.coor[2],(1,1,params.size[2]))

    # fourier space
    utility.fourier_space=Struct()
    utility.fourier_space.res=np.empty((3,3))
    utility.fourier_space.res[0]=1/(params.resolution[0]*params.size[0])
    utility.fourier_space.res[1]=1/(params.resolution[1]*params.size[1])
    utility.fourier_space.res[2]=1/(params.resolution[2]*params.size[2])
    utility.fourier_space.size=np.empty((3,3))
    utility.fourier_space.size[0]=params.size[0]
    utility.fourier_space.size[1]=params.size[1]
    utility.fourier_space.size[2]=params.size[2]
    utility.fourier_space.coor=np.empty((3,3))
    utility.fourier_space.coor[0]=np.array([i-int(params.size[0]/2) for i in range(params.size[0])])
    utility.fourier_space.coor[1]=np.array([i-int(params.size[1]/2) for i in range(params.size[1])])
    utility.fourier_space.coor[2]=np.array([i-int(params.size[2]/2) for i in range(params.size[2])])
    utility.fourier_space.coor[0]=utility.fourier_space.coor[0]*utility.fourier_space.res[0]
    utility.fourier_space.coor[1]=utility.fourier_space.coor[1]*utility.fourier_space.res[1]
    utility.fourier_space.coor[2]=utility.fourier_space.coor[2]*utility.fourier_space.res[2]
    utility.fourier_space.coor[0]=np.reshape(utility.fourier_space.coor[0],(params.size[0],1,1))
    utility.fourier_space.coor[1]=np.reshape(utility.fourier_space.coor[1],(1,params.size[1],1))
    utility.fourier_space.coor[2]=np.reshape(utility.fourier_space.coor[2],(1,1,params.size[2]))
    utility.fourier_space_coorxy=np.sqrt((utility.fourier_space.coor[0]**2) + (utility.fourier_space.coor[1]**2))

    # other
    utility.Lambda=params.wavelength
    utility.k0=1/params.wavelength
    utility.k0_nm=utility.k0*params.RI_bg
    utility.nm=params.RI_bg
    utility.kmax=params.NA/params.wavelength
    utility.NA_circle=utility.fourier_space.coorxy<utility.kmax
    utility.k3=(utility.k0_nm)**2-(utility.fourier_space.coorxy)**2

    if utility.k3 < 0:
        utility.k3 = 0

    utility.k3=np.sqrt(utility.k3)
    utility.dV=utility.image_space.res[0]*utility.image_space.res[1]*utility.image_space.res[2]
    utility.dVk=1/utility.dV
    utility.refocusing_kernel=1j*2*pi*utility.k3
    utility.cos_theta=utility.k3/utility.k0_nm

    return utility

"""
%move to the gpu the needed arrays (scalar are kept on cpu)
if use_gpu
    utility.image_space.coor{1}=gpuArray(utility.image_space.coor{1});
    utility.image_space.coor{2}=gpuArray(utility.image_space.coor{2});
    utility.image_space.coor{3}=gpuArray(utility.image_space.coor{3});
    utility.fourier_space.coor{1}=gpuArray(utility.fourier_space.coor{1});
    utility.fourier_space.coor{2}=gpuArray(utility.fourier_space.coor{2});
    utility.fourier_space.coor{3}=gpuArray(utility.fourier_space.coor{3});
    utility.fourier_space.coorxy=gpuArray(utility.fourier_space.coorxy);
    utility.NA_circle=gpuArray(utility.NA_circle);
    utility.k3=gpuArray(utility.k3);
    utility.refocusing_kernel=gpuArray(utility.refocusing_kernel);
    utility.cos_theta=gpuArray(utility.cos_theta);
end

end
"""
