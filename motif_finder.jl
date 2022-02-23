### A Pluto.jl notebook ###
# v0.18.1

using Markdown
using InteractiveUtils

# ╔═╡ 7f9ec4f2-947f-11ec-05bc-df604fbd0243
using GZip, JSON

# ╔═╡ 1d763f26-d916-4833-bad0-12624ce16072
load(typ, path) = [parse(typ, l) for l in readlines(GZip.open(path))]

# ╔═╡ cca91a1f-14b0-48c6-8cbd-ec91d45dcdfd
idxs = [parse(Int, l) for l in readlines(GZip.open("/tmp/astro.mp.idx.gz"))]

# ╔═╡ fad3f96e-ab9a-4956-b7fc-8edc3d720115
function findmotif(dists, idxs, ex)
	a = argmin(dists)
	motif_d = dists[a]
	b = idxs[a]
	if isfinite(dists[b + 1])
		# SCAMP is 0-based indexed
		(a-1, b, motif_d)
	else
		dists[a] = Inf
		findmotif(dists, idxs, ex)
	end
end

# ╔═╡ ed89e056-d62f-42b8-9328-5784773fcf24
function findmotifs(dists, idxs, k, ex)
	dists = deepcopy(dists)
	motifs = []
	for _ in 1:k
		(a, b, d) = findmotif(dists, idxs, ex)
		dists[a-ex:a+ex] .= Inf
		dists[b-ex:b+ex] .= Inf
		push!(motifs, (a, b, d))
	end
	motifs
end

# ╔═╡ 07a42c2b-e9ed-4704-99b7-6b51d0fe58e6
baseline = Dict(
	"ECG" => findmotifs(
		load(Float64, "data/ecg.mp.dists.gz"), 
		load(Int, "data/ecg.mp.idx.gz"), 10, 1000),
	"ASTRO" => findmotifs(
		load(Float64, "data/astro.mp.dists.gz"), 
		load(Int, "data/astro.mp.idx.gz"), 10, 100),
)

# ╔═╡ 4e302e0f-b529-490d-b106-435d59cb6447
open("baselines.json", "w") do f
	JSON.print(f, baseline, 2)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
GZip = "92fee26a-97fe-5a0c-ad85-20a5f3185b63"
JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"

[compat]
GZip = "~0.5.1"
JSON = "~0.21.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.0"
manifest_format = "2.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.GZip]]
deps = ["Libdl"]
git-tree-sha1 = "039be665faf0b8ae36e089cd694233f5dee3f7d6"
uuid = "92fee26a-97fe-5a0c-ad85-20a5f3185b63"
version = "0.5.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "13468f237353112a01b2d6b32f3d0f80219944aa"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
"""

# ╔═╡ Cell order:
# ╠═7f9ec4f2-947f-11ec-05bc-df604fbd0243
# ╠═1d763f26-d916-4833-bad0-12624ce16072
# ╠═cca91a1f-14b0-48c6-8cbd-ec91d45dcdfd
# ╠═fad3f96e-ab9a-4956-b7fc-8edc3d720115
# ╠═ed89e056-d62f-42b8-9328-5784773fcf24
# ╠═07a42c2b-e9ed-4704-99b7-6b51d0fe58e6
# ╠═4e302e0f-b529-490d-b106-435d59cb6447
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
