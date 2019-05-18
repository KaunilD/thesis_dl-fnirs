#### Multi-label classification in Deep Learning

1. __Binary Relevance (BR) transformation__:  
     1. Essentially, a multi-label problem is transformed into one binary problem for each label and any off-the-shelf binary classifier is applied to each of these problems individually.
     2. Multi-label literature identiﬁes that __this method is limited by the fact that dependencies between labels are not explicitly modelled__ and proposes algorithms to take these dependencies into account.
     3. The binary relevance approach (BR) does not obtain high predictive performance because it does not model dependencies between labels.
2. __*Label Powerset (LP)__:  
     1. Transforms the multi-label problem into single label problem with a single class, having the powerset as the set of values (i.e., all possible 2^L combinations)
     2. For example, if possible labels for an example were A, B, and C, the label powerset representation of this problem is a multi-class classification problem with the classes [0 0 0], [1 0 0], [0 1 0], [0 0 1], [1 1 0], [1 0 1], [0 1 1], [1 1 1] where for example [1 0 1] denotes an example where labels A and C are present and label B is absent.
3. __Classiﬁer Chain (CC)__:
     1. This method employs one classifier for each label, like BR, but the classifiers are not independent. Rather, each classifier predicts the binary relevance of each label given the input space plus the predictions of the previous classifiers (hence the chain).
     2. It combines the computational efficiency of the BR method while still being able to take the label dependencies into account for classification.
