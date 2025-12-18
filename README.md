
# sequentialknn

A rust package that uses [kiddo](https://github.com/sdd/kiddo) for a very
specific KNN problem that comes up in spatial statistics a lot. In pseudocode,
the structure is
```
tree = [...]
for j in 1:length(locations)
    compute_knn_from_past_points(tree, locations[j], k)
    add_point(tree, locations[j])
end
```
I am effectively exclusively a Julia user, and so this repository only exists to
wrap the one specific function I need into a library that can be called from
Julia. See the `julia` folder for a usage example.

