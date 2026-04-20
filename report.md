# Connecting Attacks and Testing

FGSM, PGD, and DeepXplore can be combined because they are all gradient-guided
ways of searching for model failures. In my experiment, many CIFAR-10 inputs
already produced different predictions between two ResNet50 models, while some
needed small perturbations. This suggests that adversarial attacks can be useful
seed generators for differential testing: instead of starting only from clean
test images, DeepXplore could start from FGSM or PGD examples that are already
near a model decision boundary.

The combination could also work in the other direction. Standard FGSM or PGD
usually maximizes classification loss, but DeepXplore adds another objective:
activate neurons that have not been covered before. A coverage-guided attack
could optimize a combined loss, such as misclassification loss plus a weighted
neuron coverage term. This would make the attack search for inputs that are not
only adversarial, but also behaviorally diverse inside the network. Such inputs
may expose more failure modes than repeatedly attacking the same highly
vulnerable features.

This could improve efficiency. PGD is good at quickly moving an input toward a
failure region, while neuron coverage can prevent the search from staying in one
narrow part of the model behavior space. For testing multiple models, a useful
objective would be to reduce confidence in the shared original label for one
model while increasing coverage in both models. If the models then disagree, the
input becomes a suspicious test case.

However, the methods are not always beneficial together. If the coverage metric
is too coarse, the search may maximize coverage without finding meaningful
semantic failures. Also, adversarial perturbations may create unrealistic images,
which weakens the value of the test if the goal is real-world reliability. The
best combination is therefore constrained: use adversarial gradients to speed up
the search, but keep perturbations small and inspect the generated inputs
visually.
