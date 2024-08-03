Dear new user,

Hopefully the rom package will help grasping the basics of Reduced Order Modelling in regards to feul burnup anlysis in nuclear energy.

Central to the package is the DecayChain object (called so for historical reasons). The DecayChain object accepts any kind of decay array, initial condition and all the other parameters that are needed to do a burnup analysis. 

There are three relevant python files:

* decay_chain.py --> creates the aforementioned DecayChain object and gives it methods, such as run_simulation. Furthermore includes a function to create a burnup array for an artificial decay chain.
* reduce_model.py --> functions and methods needed for using Proper Orthogonal Decomposition model reduction and assist in other methods.
* utils.py --> analysis methods to check results and utilities to help with various tasks, such as inputting a serpent burnup file.

All of the functions have descriptions. Sometimes comments are added to point out certain possible improvements.

All the best,
Jonathan Pilgram
jonathanpilgram@gmail.com
