import ML_WEIGHTS 
import ML_project_updated_predictO3 

what_do_you_want = input("Predictive Model or Weighted?")
if what_do_you_want == "Predictive Model":
	which_dataset = input("Use sensor 137, 016, or 103?")
	ML_project_updated_predictO3.Predictive_Model(
		"HW"+which_dataset+"-92218-92818_data.csv"
	) 
if what_do_you_want == "Weighted":
	tol = 1e-6
	rate = 0.001 
	datasets = ["HW137-92218-92818_O3data.csv"
				,"HW103-92218-92818_O3data.csv" 
				,"HW016-92218-92818_O3data.csv"
				]
	print('')
	print('Ranked categories for each sensor')
	for j in range(len(datasets)):
		ML_WEIGHTS.LMS(datasets[j], rate).batch_gradient_descent(tol)

	print('')