python paper/results/make-tables/json_to_latex.py paper/results/data/forward_2d paper/results/tables/project-2d-performance.tex --caption "2D \$\\rightarrow\$ 2D forward projection throughput in \$10^3\$ projections/second" --label "tab:project-2d-performance"

python paper/results/make-tables/json_to_latex.py paper/results/data/forward_3d_to_2d paper/results/tables/project-3d-to-2d-performance.tex --caption "3D \$\\rightarrow\$ 2D forward projection throughput in \$10^3\$ projections/second" --label "tab:project-3d-to-2d-performance"

python paper/results/make-tables/json_to_latex.py paper/results/data/backward_2d paper/results/tables/backproject-2d-performance.tex --caption "2D \$\\rightarrow\$ 2D backward projection throughput in \$10^3\$ projections/second" --label "tab:backproject-2d-performance"

python paper/results/make-tables/json_to_latex.py paper/results/data/backward_2d_to_3d paper/results/tables/backproject-2d-to-3d-performance.tex --caption "2D \$\\rightarrow\$ 3D backward projection throughput in \$10^3\$ projections/second" --label "tab:backproject-2d-to-3d-performance"

python paper/results/make-tables/comparison_to_latex.py paper/results/data/forward_3d_to_2d_comparison paper/results/tables/project-3d-to-2d-comparison.tex --caption "3D\$\\rightarrow\$2D forward projection in torch-projectors and torch-fourier-slice, throughput in \$10^3\$ projections/second" --label "tab:project-3d-to-2d-comparison"

python paper/results/make-tables/comparison_to_latex.py paper/results/data/backward_2d_to_3d_comparison paper/results/tables/backproject-2d-to-3d-comparison.tex --caption "2D\$\\rightarrow\$3D backward projection in torch-projectors and torch-fourier-slice, throughput in \$10^3\$ projections/second" --label "tab:backproject-2d-to-3d-comparison"

python paper/results/make-tables/oversampling_table_generator.py paper/results/data/oversampling_benchmark paper/results/tables/oversampling-table.tex --caption "Throughput depending on interpolation and padding strategy, \$10^3\$ projections/second"