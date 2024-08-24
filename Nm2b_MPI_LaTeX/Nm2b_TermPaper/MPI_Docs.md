### ToC (try2)
1. Introduction:
	1. Objective: Parallelization of the FTCS Method to solve a given PDE
	2. the given PDE is a Diffusion Equation (i.e. incompressible Navier-Stokes (Convection-Diffusion) but simplified even further to Pure-Diffusion)
	3. Interpretation on what the equation represents
		1. Graphical Representation
	4. Overview of the Project and what's coming in the document ahead
		1. Serial Method, Parallelized Method
2. Serial Method
	1. Algorithm
	2. Descretization
	3. Code and Results (Plots)
		1. Treated as a control to check the results of the Parallelized method
3. Parallelized Method
	1. Domain Decomposition with Descretization
	2. Halo and Communcation
	3. Algorithm for Communication and how the Serial Method is incorporated
	4. Code and Results (Plots)
		1. Determining values of h for even distribution between ranks
		2. Decreasing h results
		3. Amdahl's Law (Strong Plot)
4. Conclusion:
	1. Limitations and room for improvement
		1. Even distribution	
	2. Problems faced using mpi4py

---

Serial Method:
	Change the passage, give a proper introduction of the section:
		by giving an overview of the subsections [Ingredients needed to conduct the serial method]

Maybe instead of a separate Python program, we can also show the coding implementation alongside the mathematical work in each section? Then main FTCS code in its own Python Program!