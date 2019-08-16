from functools import partial

import cv2
import numpy as np

from facelib import FaceType
from interact import interact as io
from mathlib import get_power_of_two
from models import ModelBase
from nnlib import nnlib
from samplelib import *

from facelib import PoseEstimator

class AVATARModel(ModelBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                            ask_sort_by_yaw=False,
                            ask_random_flip=False,
                            ask_src_scale_mod=False)

    #override
    def onInitializeOptions(self, is_first_run, ask_override):
        #def_resolution = 256
        #if is_first_run:
        #    self.options['resolution'] = io.input_int("Resolution ( 128,256 ?:help skip:%d) : " % def_resolution, def_resolution, [128,256], help_message="More resolution requires more VRAM and time to train. Value will be adjusted to multiple of 16.")
        #else:
        #    self.options['resolution'] = self.options.get('resolution', def_resolution)

        if is_first_run or ask_override:
            def_stage = self.options.get('stage', 0)
            self.options['stage'] = io.input_int("Stage (0, 1, 2 ?:help skip:%d) : " % def_stage, def_stage, [0,1,2], help_message="Train first stage, then second. Tune batch size to maximum possible for both stages.")
        else:            
            self.options['stage'] = self.options.get('stage', 0)
            
    #override
    def onInitialize(self, batch_size=-1, **in_options):
        exec(nnlib.code_import_all, locals(), globals())
        self.set_vram_batch_requirements({6:4})
        AVATARModel.initialize_nn_functions()


        resolution = self.resolution = 224 #self.options['resolution']
        stage = self.stage = self.options['stage']
        df_res = self.df_res = 128
        in_bgr_shape = (df_res, df_res, 3)
        bgr_64_mask_shape = (df_res, df_res, 1)
        out_bgr_shape = (resolution, resolution, 3)
        bgr_t_shape = (resolution, resolution, 9)

        self.enc = modelify(AVATARModel.EncFlow())( [Input(in_bgr_shape),] )

        self.decA64 = modelify(AVATARModel.DecFlow()) ( [ Input(K.int_shape(self.enc.outputs[0])[1:]) ] )        
        self.decB64 = modelify(AVATARModel.DecFlow()) ( [ Input(K.int_shape(self.enc.outputs[0])[1:]) ] )
        self.D = modelify(AVATARModel.Discriminator() ) (Input(in_bgr_shape))     
        self.C = modelify(AVATARModel.ResNet (9, use_batch_norm=False, n_blocks=6, ngf=128, use_dropout=False))( Input(bgr_t_shape))
           
         
        if self.is_first_run():
            conv_weights_list = []
            for model in [self.enc, self.decA64, self.decB64, self.C, self.D]:
                for layer in model.layers:
                    if type(layer) == keras.layers.Conv2D:
                        conv_weights_list += [layer.weights[0]] #Conv2D kernel_weights
            CAInitializerMP ( conv_weights_list )
                
        if not self.is_first_run():
            weights_to_load = [
                [self.enc, 'enc.h5'],
                [self.decA64, 'decA64.h5'],
                [self.decB64, 'decB64.h5'],
                [self.C, 'C.h5'],
                [self.D, 'D.h5'],
            ]
            self.load_weights_safe(weights_to_load)
        
        def DLoss(labels,logits):
            return K.mean(K.binary_crossentropy(labels,logits))
            
        warped_A64 = Input(in_bgr_shape)
        real_A64 = Input(in_bgr_shape)
        real_A64m = Input(bgr_64_mask_shape)
        
        real_B64_t0 = Input(in_bgr_shape)
        real_B64_t1 = Input(in_bgr_shape)
        real_B64_t2 = Input(in_bgr_shape)        
        
        real_A64_t0 = Input(in_bgr_shape)
        real_A64m_t0 = Input(bgr_64_mask_shape)
        real_A_t0 = Input(out_bgr_shape)
        real_A64_t1 = Input(in_bgr_shape)
        real_A64m_t1 = Input(bgr_64_mask_shape)
        real_A_t1 = Input(out_bgr_shape)
        real_A64_t2 = Input(in_bgr_shape)
        real_A64m_t2 = Input(bgr_64_mask_shape)
        real_A_t2 = Input(out_bgr_shape)
        
        warped_B64 = Input(in_bgr_shape)
        real_B64 = Input(in_bgr_shape)
        real_B64m = Input(bgr_64_mask_shape)
        
        warped_A_code = self.enc (warped_A64)
        warped_B_code = self.enc (warped_B64)

        rec_A64 = self.decA64(warped_A_code)
        rec_B64 = self.decB64(warped_B_code)
        rec_AB64 = self.decA64(warped_B_code)
        
        real_A64_d = self.D( real_A64*real_A64m + (1-real_A64m)*0.5)
        real_A64_d_ones = K.ones_like(real_A64_d)        
        fake_A64_d = self.D(rec_AB64)
        fake_A64_d_ones = K.ones_like(fake_A64_d)
        fake_A64_d_zeros = K.zeros_like(fake_A64_d)

        
        def gray_pad(x):
            a = np.ones((resolution,resolution,3))*0.5
            pad = ( resolution - df_res ) // 2
            a[pad:-pad:,pad:-pad:,:] = 0
            return K.spatial_2d_padding(x, padding=((pad, pad), (pad, pad)) ) + K.constant(a, dtype=K.floatx() )
            
        def Cto3t(x):
            return Lambda ( lambda x: x[...,0:3], output_shape= ( K.int_shape(x)[1:3], 3 ) ) (x), \
                   Lambda ( lambda x: x[...,3:6], output_shape= ( K.int_shape(x)[1:3], 3 ) ) (x), \
                   Lambda ( lambda x: x[...,6:9], output_shape= ( K.int_shape(x)[1:3], 3 ) ) (x)
                   
        rec_AB_t0 = gray_pad( self.decA64 (self.enc (real_B64_t0)) )
        rec_AB_t1 = gray_pad( self.decA64 (self.enc (real_B64_t1)) )
        rec_AB_t2 = gray_pad( self.decA64 (self.enc (real_B64_t2)) )
        
        C_in_A_t0 = gray_pad(real_A64_t0*real_A64m_t0 + (1-real_A64m_t0)*0.5)
        C_in_A_t1 = gray_pad(real_A64_t1*real_A64m_t1 + (1-real_A64m_t1)*0.5)
        C_in_A_t2 = gray_pad(real_A64_t2*real_A64m_t2 + (1-real_A64m_t2)*0.5)
       
        rec_C_A_t0, rec_C_A_t1, rec_C_A_t2 = Cto3t ( self.C ( K.concatenate ( [C_in_A_t0, C_in_A_t1, C_in_A_t2] , axis=-1) ) )
        rec_C_AB_t0, rec_C_AB_t1, rec_C_AB_t2 = Cto3t( self.C ( K.concatenate ( [rec_AB_t0, rec_AB_t1, rec_AB_t2] , axis=-1) ) )

        self.G64_view = K.function([warped_A64, warped_B64],[rec_A64, rec_B64, rec_AB64])
        self.G_view = K.function([real_A64_t0, real_A64m_t0, real_A64_t1, real_A64m_t1, real_A64_t2, real_A64m_t2, real_B64_t0, real_B64_t1, real_B64_t2], [rec_C_A_t0, rec_C_A_t1, rec_C_A_t2, rec_C_AB_t0, rec_C_AB_t1, rec_C_AB_t2])
        
        if self.is_training_mode:
            loss_AB64 = K.mean(10 * dssim(kernel_size=int(df_res/11.6),max_value=1.0) ( rec_A64, real_A64*real_A64m + (1-real_A64m)*0.5) ) + \
                        K.mean(10 * dssim(kernel_size=int(df_res/11.6),max_value=1.0) ( rec_B64, real_B64*real_B64m + (1-real_B64m)*0.5) ) + 0.1*DLoss(fake_A64_d_ones, fake_A64_d )
                        
            weights_AB64 = self.enc.trainable_weights + self.decA64.trainable_weights + self.decB64.trainable_weights

            loss_C = K.mean( 10 * dssim(kernel_size=int(resolution/11.6),max_value=1.0) ( real_A_t0, rec_C_A_t0 ) ) + \
                     K.mean( 10 * dssim(kernel_size=int(resolution/11.6),max_value=1.0) ( real_A_t1, rec_C_A_t1 ) ) + \
                     K.mean( 10 * dssim(kernel_size=int(resolution/11.6),max_value=1.0) ( real_A_t2, rec_C_A_t2 ) ) #+ DLoss(fake_A_d_ones, fake_A_d )
            weights_C = self.C.trainable_weights
            
            loss_D = (DLoss(real_A64_d_ones, real_A64_d ) + \
                        DLoss(fake_A64_d_zeros, fake_A64_d ) ) * 0.5
                      
            def opt(lr=5e-5):
                return Adam(lr=lr, beta_1=0.5, beta_2=0.999, tf_cpu_mode=2 if 'tensorflow' in self.device_config.backend else 0 )

            self.AB64_train = K.function ([warped_A64, real_A64, real_A64m, warped_B64, real_B64, real_B64m], [loss_AB64], opt().get_updates(loss_AB64, weights_AB64) )
            self.C_train = K.function ([real_A64_t0, real_A64m_t0, real_A_t0,
                                        real_A64_t1, real_A64m_t1, real_A_t1, 
                                        real_A64_t2, real_A64m_t2, real_A_t2],[ loss_C ], opt().get_updates(loss_C, weights_C) )
           
            self.D_train = K.function ([warped_A64, real_A64, real_A64m, warped_B64, real_B64, real_B64m],[loss_D], opt().get_updates(loss_D, self.D.trainable_weights) )
            
            ###########
            t = SampleProcessor.Types

            generators = [
                    SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=False),
                        output_sample_types=[ {'types': (t.IMG_WARPED_TRANSFORMED, t.FACE_TYPE_FULL_NO_ALIGN, t.MODE_BGR), 'resolution':df_res},
                                              {'types': (t.IMG_TRANSFORMED, t.FACE_TYPE_FULL_NO_ALIGN, t.MODE_BGR), 'resolution':df_res},
                                              {'types': (t.IMG_TRANSFORMED, t.FACE_TYPE_FULL_NO_ALIGN, t.MODE_M), 'resolution':df_res}
                                            ] ),
                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=False),
                        output_sample_types=[ {'types': (t.IMG_WARPED_TRANSFORMED, t.FACE_TYPE_FULL_NO_ALIGN, t.MODE_BGR), 'resolution':df_res},
                                              {'types': (t.IMG_TRANSFORMED, t.FACE_TYPE_FULL_NO_ALIGN, t.MODE_BGR), 'resolution':df_res},
                                              {'types': (t.IMG_TRANSFORMED, t.FACE_TYPE_FULL_NO_ALIGN, t.MODE_M), 'resolution':df_res}
                                            ] ),
                        
                    SampleGeneratorFaceTemporal(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size, 
                        temporal_image_count=3,
                        sample_process_options=SampleProcessor.Options(random_flip=False), 
                        output_sample_types=[{'types': (t.IMG_WARPED_TRANSFORMED, t.FACE_TYPE_FULL_NO_ALIGN, t.MODE_BGR), 'resolution':df_res},#IMG_WARPED_TRANSFORMED
                                             {'types': (t.IMG_WARPED_TRANSFORMED, t.FACE_TYPE_FULL_NO_ALIGN, t.MODE_M), 'resolution':df_res},
                                             {'types': (t.IMG_SOURCE, t.NONE, t.MODE_BGR), 'resolution':resolution},
                                            ] ),
                       
                    SampleGeneratorFaceTemporal(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size, 
                        temporal_image_count=3,
                        sample_process_options=SampleProcessor.Options(random_flip=False), 
                        output_sample_types=[{'types': (t.IMG_SOURCE, t.FACE_TYPE_FULL_NO_ALIGN, t.MODE_BGR), 'resolution':df_res},
                                             {'types': (t.IMG_SOURCE, t.NONE, t.MODE_BGR), 'resolution':resolution},
                                            ] ), 
                   ]
                   
            if self.stage == 1:
                generators[2].set_active(False)
                generators[3].set_active(False)
            elif self.stage == 2:
                generators[0].set_active(False)
                generators[1].set_active(False)
            
            self.set_training_data_generators (generators)
        else:
            self.G_convert = K.function([real_B64_t0, real_B64_t1, real_B64_t2],[rec_C_AB_t1])
            
    #override , return [ [model, filename],... ]  list
    def get_model_filename_list(self):
        return [   [self.enc, 'enc.h5'],
                    [self.decA64, 'decA64.h5'],
                    [self.decB64, 'decB64.h5'],
                    [self.C, 'C.h5'],
                    [self.D, 'D.h5'],
               ]
        
    #override
    def onSave(self):
        self.save_weights_safe( [   [self.enc, 'enc.h5'],
                                    [self.decA64, 'decA64.h5'],
                                    [self.decB64, 'decB64.h5'],
                                    [self.C, 'C.h5'],
                                    [self.D, 'D.h5'],
                                 ])

    #override
    def onTrainOneIter(self, generators_samples, generators_list):
        warped_src64, src64, src64m = generators_samples[0]
        warped_dst64, dst64, dst64m = generators_samples[1]        
        
        real_A64_t0, real_A64m_t0, real_A_t0, real_A64_t1, real_A64m_t1, real_A_t1, real_A64_t2, real_A64m_t2, real_A_t2 = generators_samples[2]
        real_B64_t0, _, real_B64_t1, _, real_B64_t2, _ = generators_samples[3]

        if self.stage == 0 or self.stage == 1:
            loss,   = self.AB64_train ( [warped_src64, src64, src64m, warped_dst64, dst64, dst64m] ) 
            loss_D, = self.D_train  ( [warped_src64, src64, src64m, warped_dst64, dst64, dst64m] )
            if self.stage != 0:
                loss_C = 0
            
        if self.stage == 0 or self.stage == 2:
            loss_C1, = self.C_train ( [real_A64_t0, real_A64m_t0, real_A_t0, real_A64_t1, real_A64m_t1, real_A_t1, real_A64_t2, real_A64m_t2, real_A_t2] )
            loss_C2, = self.C_train ( [real_A64_t2, real_A64m_t2, real_A_t2, real_A64_t1, real_A64m_t1, real_A_t1, real_A64_t0, real_A64m_t0, real_A_t0] )
            loss_C = (loss_C1 + loss_C2) / 2
            if self.stage != 0:
                loss, loss_D = 0, 0
        
        return ( ('loss', loss), ('D', loss_D), ('C', loss_C) )

    #override
    def onGetPreview(self, sample):
        test_A064w  = sample[0][0][0:4]
        test_A064r  = sample[0][1][0:4]
        test_A064m  = sample[0][2][0:4]
        
        test_B064w  = sample[1][0][0:4]
        test_B064r  = sample[1][1][0:4]
        test_B064m  = sample[1][2][0:4]
        
        t_src64_0  = sample[2][0][0:4]
        t_src64m_0 = sample[2][1][0:4]
        t_src_0    = sample[2][2][0:4]
        t_src64_1  = sample[2][3][0:4]
        t_src64m_1 = sample[2][4][0:4]
        t_src_1    = sample[2][5][0:4]
        t_src64_2  = sample[2][6][0:4]
        t_src64m_2 = sample[2][7][0:4]
        t_src_2    = sample[2][8][0:4]
        
        t_dst64_0 = sample[3][0][0:4]
        t_dst_0   = sample[3][1][0:4]
        t_dst64_1 = sample[3][2][0:4]
        t_dst_1   = sample[3][3][0:4]
        t_dst64_2 = sample[3][4][0:4]
        t_dst_2   = sample[3][5][0:4]
        
        G64_view_result = self.G64_view ([test_A064r, test_B064r])
        test_A064r, test_B064r, rec_A64, rec_B64, rec_AB64 = [ x[0] for x in ([test_A064r, test_B064r] + G64_view_result)  ]
        
        sample64x4 = np.concatenate ([ np.concatenate ( [rec_B64, rec_A64], axis=1 ),
                                       np.concatenate ( [test_B064r, rec_AB64], axis=1) ], axis=0 )
                                       
        sample64x4 = cv2.resize (sample64x4, (self.resolution, self.resolution) )

        G_view_result = self.G_view([t_src64_0, t_src64m_0, t_src64_1, t_src64m_1, t_src64_2, t_src64m_2, t_dst64_0, t_dst64_1, t_dst64_2 ])
        
        t_dst_0, t_dst_1, t_dst_2, rec_C_A_t0, rec_C_A_t1, rec_C_A_t2, rec_C_AB_t0, rec_C_AB_t1, rec_C_AB_t2 = [ x[0] for x in ([t_dst_0, t_dst_1, t_dst_2, ] + G_view_result)  ]
        

        c1 = np.concatenate ( (sample64x4, rec_C_A_t0, t_dst_0, rec_C_AB_t0 ), axis=1 )
        c2 = np.concatenate ( (sample64x4, rec_C_A_t1, t_dst_1, rec_C_AB_t1 ), axis=1 )
        c3 = np.concatenate ( (sample64x4, rec_C_A_t2, t_dst_2, rec_C_AB_t2 ), axis=1 )
        
        r = np.concatenate ( [c1,c2,c3], axis=0 )
        #r = sample64x4
        return [ ('AVATAR', r ) ]

    def predictor_func (self, inp_f0, inp_f1, inp_f2):        
        feed = [ inp_f0[np.newaxis,...], inp_f1[np.newaxis,...], inp_f2[np.newaxis,...] ]
        x = self.G_convert (feed)[0]
        return np.clip ( x[0], 0, 1)

    # #override
    # def get_converter(self, **in_options):
    #     from models import ConverterImage
    #     return ConverterImage(self.predictor_func,
    #                           predictor_input_size=self.options['resolution'],
    #                           **in_options)
    #override
    def get_converter(self):
        base_erode_mask_modifier = 30
        base_blur_mask_modifier = 0

        default_erode_mask_modifier = 0
        default_blur_mask_modifier = 0

        face_type = FaceType.FULL

        from converters import ConverterAvatar
        return ConverterAvatar(self.predictor_func,
                               predictor_input_size=self.df_res)
                               
    @staticmethod
    def NLayerDiscriminator(ndf=64, n_layers=3):
        exec (nnlib.import_all(), locals(), globals())

        #use_bias = True
        #def XNormalization(x):
        #    return InstanceNormalization (axis=-1)(x)
        use_bias = False
        def XNormalization(x):
            return BatchNormalization (axis=-1)(x)
                
        XConv2D = partial(Conv2D, use_bias=use_bias)
 
        def func(x):
            f = ndf

            x = XConv2D( f, 4, strides=2, padding='same', use_bias=True)(x)
            f = min( ndf*8, f*2 )
            x = LeakyReLU(0.2)(x)

            for i in range(n_layers):
                x = XConv2D( f, 4, strides=2, padding='same')(x)               
                x = XNormalization(x)
                x = LeakyReLU(0.2)(x)
                f = min( ndf*8, f*2 )
                
            x = XConv2D( f, 4, strides=1, padding='same')(x)
            x = XNormalization(x)
            x = LeakyReLU(0.2)(x)

            return XConv2D( 1, 4, strides=1, padding='same', use_bias=True, activation='sigmoid')(x)#
        return func
      
    """  
    @staticmethod
    def Discriminator(ndf=128):
        exec (nnlib.import_all(), locals(), globals())

        #use_bias = True
        #def XNormalization(x):
        #    return InstanceNormalization (axis=-1)(x)
        use_bias = False
        def XNormalization(x):
            return BatchNormalization (axis=-1)(x)

        XConv2D = partial(Conv2D, use_bias=use_bias)

        def func(input):
            b,h,w,c = K.int_shape(input)

            x = input

            x = XConv2D( ndf, 4, strides=2, padding='same', use_bias=True)(x)
            x = LeakyReLU(0.2)(x)

            x = XConv2D( ndf*2, 4, strides=2, padding='same')(x)
            x = XNormalization(x)
            x = LeakyReLU(0.2)(x)

            x = XConv2D( ndf*4, 4, strides=2, padding='same')(x)
            x = XNormalization(x)
            x = LeakyReLU(0.2)(x)
            
            x = XConv2D( ndf*8, 4, strides=2, padding='same')(x)
            x = XNormalization(x)
            x = LeakyReLU(0.2)(x)
            
            return XConv2D( 1, 4, strides=1, padding='same', use_bias=True, activation='sigmoid')(x)#
        return func
    """
    @staticmethod
    def Discriminator(ndf=128):
        exec (nnlib.import_all(), locals(), globals())

        use_bias = True
        def XNormalization(x):
            return InstanceNormalization (axis=-1)(x)
        #use_bias = False
        #def XNormalization(x):
        #    return BatchNormalization (axis=-1)(x)

        XConv2D = partial(Conv2D, use_bias=use_bias)

        def func(input):
            b,h,w,c = K.int_shape(input)

            x = input

            x = XConv2D( ndf, 4, strides=2, padding='same', use_bias=True)(x)
            x = LeakyReLU(0.2)(x)

            x = XConv2D( ndf*2, 4, strides=2, padding='same')(x)
            x = XNormalization(x)
            x = LeakyReLU(0.2)(x)

            x = XConv2D( ndf*4, 4, strides=2, padding='same')(x)
            x = XNormalization(x)
            x = LeakyReLU(0.2)(x)
            
            x = XConv2D( ndf*8, 4, strides=2, padding='same')(x)
            x = XNormalization(x)
            x = LeakyReLU(0.2)(x)
            
            return XConv2D( 1, 4, strides=1, padding='same', use_bias=True, activation='sigmoid')(x)#
        return func
    
    @staticmethod
    def EncFlow(padding='zero', **kwargs):
        exec (nnlib.import_all(), locals(), globals())

        use_bias = False
        def XNorm(x):
            return BatchNormalization (axis=-1)(x)
        XConv2D = partial(Conv2D, padding=padding, use_bias=use_bias)

        def downscale (dim):
            def func(x):
                return LeakyReLU(0.1)( Conv2D(dim, 5, strides=2, padding='same')(x))
            return func

        def upscale (dim):
            def func(x):
                return SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(dim * 4, 3, strides=1, padding='same')(x)))
            return func

               
        def func(input):
            x, = input
            b,h,w,c = K.int_shape(x)
            
            dim_res = w // 16
            
            x = downscale(64)(x)
            x = downscale(128)(x)
            x = downscale(256)(x)
            x = downscale(512)(x)

            x = Dense(512)(Flatten()(x))
            x = Dense(dim_res * dim_res * 512)(x)
            x = Reshape((dim_res, dim_res, 512))(x) 
            x = upscale(512)(x)   
            return x
            
        return func

    @staticmethod
    def DecFlow(output_nc=3, **kwargs):
        exec (nnlib.import_all(), locals(), globals())

        ResidualBlock = AVATARModel.ResidualBlock
        upscale = AVATARModel.upscale
        to_bgr = AVATARModel.to_bgr

        def func(input):
            x = input[0]
            
            x = upscale(512)(x)
            x = upscale(256)(x)
            x = upscale(128)(x)
            return to_bgr(output_nc) (x)

        return func
    """    
    @staticmethod
    def CNet(output_nc, use_batch_norm, ngf=64, n_blocks=6, use_dropout=False):
        exec (nnlib.import_all(), locals(), globals())

        if not use_batch_norm:
            use_bias = True
            def XNormalization(x):
                return InstanceNormalization (axis=-1)(x)
        else:
            use_bias = False
            def XNormalization(x):
                return BatchNormalization (axis=-1)(x)
                
        XConv2D = partial(Conv2D, padding='same', use_bias=use_bias)
        XConv2DTranspose = partial(Conv2DTranspose, padding='same', use_bias=use_bias)
        
        def ResnetBlock(dim, use_dropout=False):
            def func(input):
                x = input

                x = XConv2D(dim, 3, strides=1)(x)
                x = XNormalization(x)
                x = ReLU()(x)

                if use_dropout:
                    x = Dropout(0.5)(x)

                x = XConv2D(dim, 3, strides=1)(x)
                x = XNormalization(x)
                x = ReLU()(x)
                return Add()([x,input])
            return func
                
        def preprocess(target_res):
            def func(input):
                inp_shape = K.int_shape (input[0])
                t_len = len(input)
                total_ch = 0
                for i in range(t_len):
                    total_ch += K.int_shape (input[i])[-1]
                
                K.concatenate ( input, axis=-1) )
                import code
                c ode.interact(local=dict(globals(), **locals()))
                
                x_shape = K.int_shape(x)[1:]
                
                pad = (target_res - x_shape[0]) // 2
                
                a = np.ones((target_res,target_res,3))*0.5
                a[pad:-pad:,pad:-pad:,:] = 0
                return K.spatial_2d_padding(x, padding=((pad, pad), (pad, pad)) ) + K.constant(a, dtype=K.floatx() )
            return func

        def func(input):
            inp_shape = K.int_shape (input[0])
            t_len = len(input)
            total_ch = 0
            for i in range(t_len):
                total_ch += K.int_shape (input[i])[-1]
                
            x = Lambda ( preprocess(128) , output_shape=(inp_shape[1], inp_shape[2], total_ch)  ) (input)

            x = ReLU()(XNormalization(XConv2D(ngf, 7, strides=1)(x)))

            x = ReLU()(XNormalization(XConv2D(ngf*2, 3, strides=2)(x)))
            x = ReLU()(XNormalization(XConv2D(ngf*4, 3, strides=2)(x)))

            for i in range(n_blocks):
                x = ResnetBlock(ngf*4, use_dropout=use_dropout)(x)

            x = ReLU()(XNormalization(XConv2DTranspose(ngf*2, 3, strides=2)(x)))
            x = ReLU()(XNormalization(XConv2DTranspose(ngf  , 3, strides=2)(x)))

            x = XConv2D(output_nc, 7, strides=1, activation='sigmoid', use_bias=True)(x)

            return x

        return func
    """    
    @staticmethod
    def ResNet(output_nc, use_batch_norm, ngf=64, n_blocks=6, use_dropout=False):
        exec (nnlib.import_all(), locals(), globals())

        if not use_batch_norm:
            use_bias = True
            def XNormalization(x):
                return InstanceNormalization (axis=-1)(x)
        else:
            use_bias = False
            def XNormalization(x):
                return BatchNormalization (axis=-1)(x)
                
        XConv2D = partial(Conv2D, padding='same', use_bias=use_bias)
        XConv2DTranspose = partial(Conv2DTranspose, padding='same', use_bias=use_bias)

        def func(input):


            def ResnetBlock(dim, use_dropout=False):
                def func(input):
                    x = input

                    x = XConv2D(dim, 3, strides=1)(x)
                    x = XNormalization(x)
                    x = ReLU()(x)

                    if use_dropout:
                        x = Dropout(0.5)(x)

                    x = XConv2D(dim, 3, strides=1)(x)
                    x = XNormalization(x)
                    x = ReLU()(x)
                    return Add()([x,input])
                return func

            x = input

            x = ReLU()(XNormalization(XConv2D(ngf, 7, strides=1)(x)))

            x = ReLU()(XNormalization(XConv2D(ngf*2, 3, strides=2)(x)))
            x = ReLU()(XNormalization(XConv2D(ngf*4, 3, strides=2)(x)))
            
            x = ReLU()(XNormalization(XConv2D(ngf*4, 3, strides=2)(x)))

            for i in range(n_blocks):
                x = ResnetBlock(ngf*4, use_dropout=use_dropout)(x)
            
            x = ReLU()(XNormalization(XConv2DTranspose(ngf*4, 3, strides=2)(x)))
            
            x = ReLU()(XNormalization(XConv2DTranspose(ngf*2, 3, strides=2)(x)))
            x = ReLU()(XNormalization(XConv2DTranspose(ngf  , 3, strides=2)(x)))

            x = XConv2D(output_nc, 7, strides=1, activation='sigmoid', use_bias=True)(x)

            return x

        return func
        
    @staticmethod
    def initialize_nn_functions():
        exec (nnlib.import_all(), locals(), globals())

        class ResidualBlock(object):
            def __init__(self, filters, kernel_size=3, padding='zero', **kwargs):
                self.filters = filters
                self.kernel_size = kernel_size
                self.padding = padding

            def __call__(self, inp):
                x = inp
                x = Conv2D(self.filters, kernel_size=self.kernel_size, padding=self.padding)(x)
                x = LeakyReLU(0.2)(x)
                x = Conv2D(self.filters, kernel_size=self.kernel_size, padding=self.padding)(x)
                x = Add()([x, inp])
                x = LeakyReLU(0.2)(x)
                return x
        AVATARModel.ResidualBlock = ResidualBlock

        def downscale (dim, padding='zero', act='', **kwargs):
            def func(x):
                return LeakyReLU(0.2) (Conv2D(dim, kernel_size=5, strides=2, padding=padding)(x))
            return func
        AVATARModel.downscale = downscale

        def upscale (dim, padding='zero', norm='', act='', **kwargs):
            def func(x):
                return SubpixelUpscaler()( LeakyReLU(0.2)(Conv2D(dim * 4, kernel_size=3, strides=1, padding=padding)(x)))
            return func
        AVATARModel.upscale = upscale

        def to_bgr (output_nc, padding='zero', **kwargs):
            def func(x):
                return Conv2D(output_nc, kernel_size=5, padding=padding, activation='sigmoid')(x)
            return func
        AVATARModel.to_bgr = to_bgr
        
Model = AVATARModel

""" 
def BCELoss(logits, ones):
    if ones:
        return K.mean(K.binary_crossentropy(K.ones_like(logits),logits))
    else:
        return K.mean(K.binary_crossentropy(K.zeros_like(logits),logits))

def MSELoss(labels,logits):
    return K.mean(K.square(labels-logits))

def DLoss(labels,logits):
    return K.mean(K.binary_crossentropy(labels,logits))

def MAELoss(t1,t2):
    return dssim(kernel_size=int(resolution/11.6),max_value=2.0)(t1+1,t2+1 )
    return K.mean(K.abs(t1 - t2) )
"""