# gray

- Petit ray tracer implémenté pendant les vacances
- Juste la diffusion de lumière, pas de réflexion ou réfraction
- Support des sphères et des triangles
- Je voulais tester une optimisation que j'avais en tête sur un certain type de
scène simple, qui consiste à calculer l'aire angulaire exacte que chaque sphère
ou triangle représente en se plaçant sur un point donné sur lequel on cherche à
calculer la lumière indirecte (option
[aim_rays](https://github.com/greg904/gray/blob/46193beb35607a571cfabcbc638a47287a831f8d/src/ray_trace.rs#L190))
- L'optimisation semble marcher, mais seulement sur des cas très simples, et je
n'ai pas assez testé.
- Puis j'ai implémenté un rasterizer CPU avec la même scène pour comprendre la
rasterization, mais il manque le shading et la rasterization d'une sphère est
lente car je n'ai pas trouvé d'algorithme efficace.
