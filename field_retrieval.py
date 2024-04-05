from fundamentals.matlab_class import Struct
import numpy as np
from numpy import fft2, ifft2, fftshift, ifftshift

class Field_Retrieval(self):
    def __init__(self):
        self.parameters = Struct()
        self.utility = Struct()

        self.parameters = BASIC_OPTICAL_PARAMETER() # Struct class

        # SIMULATION PARAMETERS
        self.parameters.resolution_image=np.array([1, 1])*0.1
        self.parameters.resolution = np.array([1,1])*0.1
        self.parameters.use_abbe_correction=True
        self.parameters.cutout_portion=1/3
        self.parameters.other_corner=False; #false if the field is in another corner of the image
        self.parameters.conjugate_field=False
        self.parameters.use_GPU = True
        
        if init_params is not None:
            self.parameters=update_struct(self.parameters, init_params)

    def check_input(h, input_field, output_field, ROI):
        if input_field.shape != output_field.shape:
            raise ValueError('Background and sample field must be of same size');
    
        if input_field.shape[0] != input_field.shape[1]:
            raise ValueError('the image must be a square')
        
        if h.parameters.resolution_image[0] != h.parameters.resolution_image[1]:
            raise ValueError('please enter an isotropic resolution for the image')
        
        if h.parameters.resolution[0] != h.parameters.resolution[1]:
            raise ValueError('please enter an isotropic resolution for the output image')
    
    def crop_rectangular_image(h, input_file, output_file, ROI):
        temp_input_field = fftshift(fft2(ifftshift(input_field)))
        temp_output_field=fftshift(fft2(ifftshift(output_field)))

        # 1 center the field in the fourier space
        delete_band_1=[round(temp_input_field.shape[0]*h.parameters.cutout_portion), temp_input_field.shape[0]]
        delete_band_1=[round(temp_input_field.shape[1]*h.parameters.cutout_portion), temp_input_field.shape[1]]
        if h.parameters.other_corner:
            delete_band_2=[1,round(input_field0.shape[1])*(1-h.parameters.cutout_portion)]

        normal_bg=input_field0(:,:,1);
        normal_bg(delete_band_1,:,:)=0;
        normal_bg(:,delete_band_2,:)=0;

        [center_pos_1,center_pos_2]=find(abs(normal_bg)==max(abs(normal_bg(:))));

        input_field0=fftshift(fftshift(circshift(input_field0,[1-center_pos_1,1-center_pos_2,0]),1),2);
        output_field0=fftshift(fftshift(circshift(output_field0,[1-center_pos_1,1-center_pos_2,0]),1),2);

        %2 match to the resolution
        resolution0 = h.parameters.resolution;
        h.parameters.resolution(1)=h.parameters.resolution_image(1);
        h.parameters.resolution(2)=h.parameters.resolution_image(2);
        h.parameters.size(1)=size(input_field0,1);
        h.parameters.size(2)=size(input_field0,2);
        h.utility=DERIVE_OPTICAL_TOOL(h.parameters);
        input_field0=input_field0.*h.utility.NA_circle;
        output_field0=output_field0.*h.utility.NA_circle;
        input_field0=fftshift(ifft2(ifftshift(input_field0)));
        output_field0=fftshift(ifft2(ifftshift(output_field0)));
        retPhase=angle(output_field0./input_field0);
        retPhase=gather(unwrapp2_gpu(gpuArray(single(retPhase))));
        if h.parameters.conjugate_field
            retPhase = -retPhase;
        end
        retPhase = PhiShiftMS(retPhase,1,1);
        while true
            close all
            figure, imagesc(retPhase, [0 max(retPhase(:))]), axis image, colormap gray, colorbar
            title('Choose square ROI')
            r = drawrectangle;
            ROI = r.Position;
            ROI = [round(ROI(2)) round(ROI(2))+round(ROI(3)) round(ROI(1)) round(ROI(1))+round(ROI(3)) ];
            close all
            figure, imagesc(max(squeeze(retPhase(ROI(1):ROI(2), ROI(3):ROI(4),:,:)),[],3), [0 max(retPhase(:))]), axis image, colormap gray
            satisfied = input('Satisfied? 1/0: ');
            if satisfied
                close;
                break;
            end
        end
        input_field = input_field(ROI(1):ROI(2), ROI(3):ROI(4),:,:,:);
        output_field = output_field(ROI(1):ROI(2), ROI(3):ROI(4),:,:,:);
        h.parameters.resolution = resolution0;

    def get_fields(self, bg_file, sp_file, ROI):
        h = self
        input_field=loadTIFF(bg_file)
        output_field=loadTIFF(sp_file)
        
        # Input Check
        self.check_input(h, input_file, output_field, ROI)

                    
    