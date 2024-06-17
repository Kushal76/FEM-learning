#!/usr/bin/env python
# coding: utf-8

# In[51]:


#Seismic Analysis of a three story residential building
#Seismic Weight Calculation
no_mass = 3
weight_floor1 = 1229.10 # kN
weight_floor2 = 1167.001 #kN
weight_floor3 = 611.726 #kN


# In[52]:


weight_variables = [weight_floor1, weight_floor2, weight_floor3]
seismic_weight = sum(weight_variables)


# In[53]:


modulus_elasticity = 22360679.77 #kN
moi_column = 0.002133333 # m4
stiffness = {'k1': 296736.9534, 'k2': 296736.9534, 'k3': 247280.7945}


# In[54]:


import numpy as np
stiffness_matrix = np.array ([[stiffness['k1']+stiffness['k2'], -stiffness['k2'], 0],
    [-stiffness['k2'], stiffness['k2']+stiffness['k3'],-stiffness['k3'] ], 
      [0, -stiffness['k3'], stiffness['k3'] ]]) #kN/m


# In[55]:


mass_m1 = (weight_floor1*1000)/9.81 #kg
mass_m2 = (weight_floor2*1000)/9.81 #kg
mass_m3 = (weight_floor3*1000)/9.81 #kg


# In[56]:


mass_matrix = np.array([
    [mass_m1, 0, 0],
    [0, mass_m2, 0],
    [0, 0, mass_m3]
]) #kg


# In[59]:


print('Stiffness matrix:')
print(stiffness_matrix)
print('\nMass matrix:')
print(mass_matrix)


# In[60]:


from sympy import symbols, Matrix
w,a = symbols('w a')


# In[61]:


#determinant |k-w^2(m)|; w^2 = a
var_det = Matrix ([
    [(stiffness_matrix[0,0]/100)-(a*mass_matrix[0,0]/100000), (stiffness_matrix[0,1]/100), 0],
    [(stiffness_matrix[1,0]/100), (stiffness_matrix[1,1]/100)-(a*mass_matrix[1,1]/100000), (stiffness_matrix[1,2]/100)],
    [0, (stiffness_matrix[2,1]/100), (stiffness_matrix[2,2]/100)-(a*mass_matrix[2,2]/100000)]
])


# In[62]:


print('\n[[k]-w^2[m]] :')
print(var_det)


# In[63]:


#determinant of |k-w^2(m)| in the form of cubic equation
det_eq = var_det.det()
print('\nDeterminant |[k]-w^2[m]|:')
print(det_eq)


# In[64]:


cubic_equation = det_eq
p = det_eq.coeff(a, 3)
q = det_eq.coeff(a, 2)
r = det_eq.coeff(a, 1)
s = det_eq.coeff(a, 0)


# In[65]:


def coeff_cubic (coeff):
    print(coeff)
print('\nCoefficients of above determinant equation are:')
coeff_cubic(p)
coeff_cubic(q)
coeff_cubic(r)
coeff_cubic(s)


# In[66]:


#Finding the value of a or w^2
coefficients = [p, q, r, s]
roots = np.roots(coefficients)
print('\nThe value of w^2 are:')
print(roots)


# In[67]:


#Calculation of frequency of building in rad/sec sorted in order of modes 1, 2, 3
import math
a_value = [roots[0], roots[1], roots[2]]
a_value.reverse()
frequency_w = []
for x in a_value:
    y = math.sqrt(x)
    frequency_w.append(y)  
print('\nThe frequency of building in mode 1, mode 2, mode 3 respectively are:')
print(frequency_w)


# In[68]:


#Calculation of time period of building sorted in order of modes 1, 2, 3.
time_period = []
for freq in frequency_w:
    time = (2*3.14)/freq
    time_period.append(time)
print('\nThe time period of building in mode 1, mode 2, mode 3 respectively are:')
print (time_period)


# In[69]:


frequency_time = {
                   'Mode': [1, 2, 3],
                   'Frequency': [frequency_w[0], frequency_w[1], frequency_w[2]],
                   'Unit_F': ['rad/sec', 'rad/sec', 'rad/sec'],
                 'Time Period': [time_period[0], time_period[1], time_period[2]],
                 'Unit_T': ['sec', 'sec', 'sec']
}


# In[70]:


import pandas as pd
print('\nFrequency and time period of building are summarized in the table given below:')
print(pd.DataFrame(frequency_time))


# In[71]:


e01 = np.array([
    [var_det[0,2]],
    [var_det[1,2]]
])
e00 = np.array([
    [var_det[0,0], var_det[0,1]],
    [var_det[1,0], var_det[1,1]]
])
print('\nmatrix e01:')
print(e01)


# In[72]:


# A function used to determine inverse matrix to be used in calculation of mode shapes values.
def int_e00(freq2): #freq2 = value of w^2
    a = freq2
    int_det = np.array([
       [(stiffness_matrix[0,0]/100)-(a*mass_matrix[0,0]/100000), (stiffness_matrix[0,1]/100), 0],
    [(stiffness_matrix[1,0]/100), (stiffness_matrix[1,1]/100)-(a*mass_matrix[1,1]/100000), (stiffness_matrix[1,2]/100)],
    [0, (stiffness_matrix[2,1]/100), (stiffness_matrix[2,2]/100)-(a*mass_matrix[2,2]/100000)]  
    ])
    e00 = np.array([
    [int_det[0,0], int_det[0,1]],
    [int_det[1,0], int_det[1,1]]
    ])
    return np.linalg.inv(e00)


# In[73]:


#inv_e00 is the variable used for inverse of matrix E00
#For mode 1:
inv_e00_1 = int_e00(a_value[0])
z = -1
matmul_1 = np.matmul(inv_e00_1, e01)
mode_shape_coeff_1 = np.dot(z, matmul_1) #mode shape coefficients for mode 1 in order of floor 1 and 2.
print('\nMode shape coefficients of floor 1 and 2 for mode 1 respectively:')
print (mode_shape_coeff_1)


# In[74]:


#For mode 1:
inv_e00_2 = int_e00(a_value[1])
matmul_2 = np.matmul(inv_e00_2, e01)
mode_shape_coeff_2 = np.dot(z, matmul_2)
print('\nMode shape coefficients of floor 1 and 2 for mode 2 respectively:')
print(mode_shape_coeff_2)


# In[75]:


#For mode 3
inv_e00_3 = int_e00(a_value[2])
matmul_3 = np.matmul(inv_e00_3, e01)
mode_shape_coeff_3 = np.dot(z, matmul_3)
print('\nMode shape coefficients of floor 1 and 2 for mode 3 respectively:')
print(mode_shape_coeff_3)


# In[76]:


mode_shape_coeff = {
    'floor': [3, 2, 1],
    'mode 1': [1, mode_shape_coeff_1[1,0], mode_shape_coeff_1[0,0]],
    'mode 2': [1, mode_shape_coeff_2[1,0], mode_shape_coeff_2[0,0]],
    'mode 3': [1, mode_shape_coeff_3[1,0], mode_shape_coeff_3[0,0]]
}


# In[77]:


print('\nThe summary of mode shape coefficients of different modes is summarized below:')
print(pd.DataFrame(mode_shape_coeff))


# In[78]:


# From NBC 105: 2020
seis_zoning_factor = 0.35   # seismic zoning factor
imp_factor = 1              # importance factor
duct_factor = 4             # ductility factor
overstr_factor = 1.5        # overstrength factor


# In[79]:


def response_spectrum(a1, modeshape):
    spectral_shape_factor = a1
    elastic_site_spectra = imp_factor*seis_zoning_factor*spectral_shape_factor
    base_shear_coeff = elastic_site_spectra/(duct_factor*overstr_factor)
    mode_shape = []
    mode_shapesqr = []
    for a2 in modeshape:
        mode_shape.append(a2)
        a2_sqr = a2*a2
        mode_shapesqr.append(a2_sqr)
    effective_dict = {
        'weight_mode': [(weight_floor3*mode_shape[0]), (weight_floor2*mode_shape[1]), (weight_floor1*mode_shape[2])],
        'weight_modesqr': [(weight_floor3*mode_shapesqr[0]), (weight_floor2*mode_shapesqr[1]), (weight_floor1*mode_shapesqr[2])]
        }
    sum_weight_mode = sum(effective_dict['weight_mode'])
    sum_weight_modesqr = sum(effective_dict['weight_modesqr'])
    sum_weightmode_sqr = sum_weight_mode*sum_weight_mode
    effective_mod_gravity_load = sum_weightmode_sqr/sum_weight_modesqr
    base_shear = base_shear_coeff*effective_mod_gravity_load
    
    #Story Shear and Story Force
    story_shear = []
    story_force = []
    i=0
    for a4 in effective_dict['weight_mode']:
        f = (a4*base_shear)/sum_weight_mode
        story_force.append(f)
        if i == 0:
            s = f
            i=i+1
            story_shear.append(s)
        else: 
            s=story_shear[i-1]+f
            story_shear.append(s)
            i=i+1
    return base_shear, story_shear, story_force


# In[80]:


response_spectrum_mode1 = response_spectrum(2.5, mode_shape_coeff['mode 1'])
base_shear_mode1 = response_spectrum_mode1[0]
story_force_mode1 = response_spectrum_mode1[2]
story_shear_mode1 = response_spectrum_mode1[1]
print('\nBase shear for mode 1 is:')
print(base_shear_mode1)
print('\nStory force in story 3, 2, 1 respectively for mode 1:')
print(story_force_mode1)
print('\nStory Shear in story 3, 2, 1 respectively for mode 1:')
print(story_shear_mode1)


# In[81]:


response_spectrum_mode2 = response_spectrum(2.4, mode_shape_coeff['mode 2'])
base_shear_mode2 = response_spectrum_mode2[0]
story_force_mode2 = response_spectrum_mode2[2]
story_shear_mode2 = response_spectrum_mode2[1]
print('\nBase shear for mode 2 is:')
print(base_shear_mode2)
print('\nStory force in story 3, 2, 1 respectively for mode 2:')
print(story_force_mode2)
print('\nStory Shear in story 3, 2, 1 respectively for mode 2:')
print(story_shear_mode2)


# In[82]:


response_spectrum_mode3 = response_spectrum(2.1, mode_shape_coeff['mode 3'])
base_shear_mode3 = response_spectrum_mode3[0]
story_force_mode3 = response_spectrum_mode3[2]
story_shear_mode3 = response_spectrum_mode3[1]
print('\nBase shear for mode 3 is:')
print(base_shear_mode3)
print('\nStory force in story 3, 2, 1 respectively for mode 3:')
print(story_force_mode3)
print('\nStory Shear in story 3, 2, 1 respectively for mode 3:')
print(story_shear_mode3)


# In[83]:


response_spectrum_dict1 = {
    'Story': [3, 2, 1],
    'Story Forces': response_spectrum_mode1[2],
    'Story Shear': response_spectrum_mode1[1],
    'Base Shear': [0, 0, response_spectrum_mode1[0]],
}
response_spectrum_dict2 = {
    'Story': [3, 2, 1],
    'Story Forces': response_spectrum_mode2[2],
    'Story Shear': response_spectrum_mode2[1],
    'Base Shear': [0, 0, response_spectrum_mode2[0]],
}
response_spectrum_dict3 = {
    'Story': [3, 2, 1],
    'Story Forces': response_spectrum_mode3[2],
    'Story Shear': response_spectrum_mode3[1],
    'Base Shear': [0, 0, response_spectrum_mode3[0]]
}
print('\nThe summary of story forces, story shear and base shear in different modes is:\n')
print('Mode 1:')
print(pd.DataFrame(response_spectrum_dict1))
print('\nMode 2:')
print(pd.DataFrame(response_spectrum_dict2))
print('\nMode 3:')
print(pd.DataFrame(response_spectrum_dict3))


# In[84]:


res_story_shear = []
i=0 
j=0 
k=0
while i<3 and k<3 and j<3:
    b1=story_shear_mode1[i] 
    c1=story_shear_mode2[j]
    d1=story_shear_mode3[k]
    sum_s = (b1*b1) + (c1*c1) + (d1*d1)
    root_sum_s = math.sqrt(sum_s)
    res_story_shear.append(root_sum_s)
    i+=1
    j+=1
    k+=1
print('\nResultant story shear after applying SRSS method:')
print(res_story_shear)


# In[85]:


sum_base_shearsqr = pow(base_shear_mode1,2)+pow(base_shear_mode2,2)+pow(base_shear_mode3,2)
res_base_shear = math.sqrt(sum_base_shearsqr)
print('\nResultant base shear after applying SRSS method:')
print(res_base_shear)


# In[86]:


res_story_forces = []
i=0 
while i<3:
    if i==0:
        force = res_story_shear[i]
    else:
        force = res_story_shear[i] - res_story_shear[i-1]
    i += 1
    res_story_forces.append(force)
print('\nResultant story forces after SRSS method:')
print(res_story_forces)


# In[87]:


res_response_spectrum = {
    'Story': [3, 2, 1],
    'Story Forces': res_story_forces,
    'Story Shear': res_story_shear,
     'Base Shear': ['-', res_base_shear, '-']
}
print('\nThe summary of resultant story forces, story shear, and base shear after applying SRSS method:\n')
print(pd.DataFrame(res_response_spectrum))

