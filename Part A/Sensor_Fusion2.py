�
f�X_c           @   s�   d  Z  d d l Z d d l j Z d d l m Z m Z m Z m	 Z	 d d l
 Z
 d d l Z d �  Z d �  Z d �  Z d �  Z e �  d S(   s    Code setup. i����N(   t   loadtxtt   onest   zerost   linspacec          C   s@   t  j �  \ }  } t j �  \ } } } } |  | | | | | f S(   N(   t   IR3_PSMt   calibrationt	   sonar_PSM(   t   varV_IR3t   kIRt   varV_S1t   varV_S2t   kS1t   kS2(    (    sC   P:\2020 3rd Pro\ENMT482\Autonomous_Robotics\Part A\Sensor_Fusion.pyt   calibrate_Sensors   s    c
         C   sg   t  j | | | | | |	 � \ }
 } } } t j | |  | | � \ } } t | | | |
 | | � } | S(   N(   R   t	   MLE_sonarR   t   linear_ML_IRt   BLUE(   t   zIRt   zS1t   zS2t   x0R   R   R   R   R	   R
   t   xhat_S1t   xhat_S2t   varX_S1t   varX_S2t   xhat_IRt   varX_IR3t   xhat_fusion(    (    sC   P:\2020 3rd Pro\ENMT482\Autonomous_Robotics\Part A\Sensor_Fusion.pyt   fuseMLE_sensors   s    *c         C   sT   d d |  d | d | } d |  | | d | | | d | | | } | S(   Ni   (    (   t   varS1t   varS2t   varIR3t   S1_xt   S2_xt   IR3_xt   BLUE_VarR   (    (    sC   P:\2020 3rd Pro\ENMT482\Autonomous_Robotics\Part A\Sensor_Fusion.pyR      s    2c          C   s"  t  d d d d d �}  |  j \
 } } } } } } } } }	 }
 t �  \ } } } } } } d } t | � } g  } xr t | � D]d } | | } |	 | } |
 | } t | | | | | | | | | | �
 } | j | � t | � } q Wt j	 �  t j
 | | d � t j
 | | � t j �  d  S(   Ns   Part A/calibration.csvt	   delimitert   ,t   skiprowsi   g      @t   ro(   R    t   TR   t   lent   rangeR   t   appendt   floatt   pltt   figuret   plott   show(   t   datat   indext   timet   range_t   velocity_commandt   raw_ir1t   raw_ir2t   raw_ir3t   raw_ir4t   sonar1t   sonar2R   R   R	   R
   R   R   R   t   Nt   X_hat_fusiont   nR   R   R   R   (    (    sC   P:\2020 3rd Pro\ENMT482\Autonomous_Robotics\Part A\Sensor_Fusion.pyt   test_Sensor_Fusion(   s"    '


'
(   t   __doc__t   numpyt   npt   matplotlib.pyplott   pyplotR-   R    R   R   R   R   R   R   R   R   R?   (    (    (    sC   P:\2020 3rd Pro\ENMT482\Autonomous_Robotics\Part A\Sensor_Fusion.pyt   <module>   s   "					