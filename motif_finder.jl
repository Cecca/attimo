### A Pluto.jl notebook ###
# v0.18.1

using Markdown
using InteractiveUtils

# ╔═╡ 7f9ec4f2-947f-11ec-05bc-df604fbd0243
using GZip, JSON

# ╔═╡ 1d763f26-d916-4833-bad0-12624ce16072
load(typ, path) = [parse(typ, l) for l in readlines(GZip.open(path))]

# ╔═╡ 8f771188-5782-48f8-b894-fdba7096b599
actual = open("actual.json") do f
	entries = JSON.parse(f)
	out = Dict()
	for entry in entries
		out[entry["dataset"]] = [
			((m["a"], m["b"]), m["dist"])
			for m in entry["motif_pairs"]
		]
	end
	out
end

# ╔═╡ 27767505-36fc-49af-9d8c-77fadeee41fe
dists = load(Float64, "data/freezer.mp.dists.gz")

# ╔═╡ 1432d42f-feb5-43ef-8481-f69fd9ae9a8a
idx = load(Int, "data/freezer.mp.idx.gz")

# ╔═╡ 865014a5-4183-4999-b6d4-83e01430a855
sum(dists .== Inf)

# ╔═╡ 71406a9f-2376-4c0a-b3d7-a20c841cceeb
dists[1825969+1]

# ╔═╡ a777e6dc-31f3-4044-802a-01f30aa93e75
dists[1993859+1], idx[1993859+1]

# ╔═╡ 07275457-b76b-43d8-9fa0-f0c813eec88a
1834102 - 1825969

# ╔═╡ 01d215d5-bc6f-4d89-8789-b4fb5704c90e
function overlaps(p1, p2, ex)
	ov(x, y) = abs(x - y) <= ex
	ov(p1[1], p2[1]) || ov(p1[1], p2[2]) || ov(p1[2], p2[1])|| ov(p1[2], p2[2])
end

# ╔═╡ 3cea46f9-817c-4472-ab57-bb9e9be943b9
function pushtop!(arr, a, b, dist, ex)
	log = a == 3815626
	i = 1
	while i <= length(arr) && arr[i][2] <= dist
		if overlaps((a, b), arr[i][1], ex)
			if log
				@info "Excluding $(b) because it overlaps with $(arr[i])"
			end
			return
		end
		i += 1
	end
	insert!(arr, i, ((a,b), dist))
	i = i + 1
	while i <= length(arr)
		if overlaps((a, b), arr[i][1], ex)
			popat!(arr, i)
		else
			i += 1
		end
	end
end

# ╔═╡ 8845a7e9-288e-468a-9656-b938774ec08d
function findmotifs2(dists, idx, k, ex; top=[])
	vec = [(d, a, b) for (d, a, b) in zip(dists, 0:length(dists)-1, idx)
	       if a < b]
	ordered = sort(vec; by = x -> x[1])
	for (d, a, b) in ordered
		if length(top) >= k
			return top
		end
		pushtop!(top, a, b, d, ex)
	end
	top
end

# ╔═╡ 52f5be68-d380-4fb2-8030-75db1186722f
humany = findmotifs2(
		load(Float64, "data/HumanY.mp.dists.gz"), 
		load(Int, "data/HumanY.mp.idx.gz"), 10, 18000;
		top=actual["HumanY"])

# ╔═╡ 5dbfa9d3-a6a0-4be3-a3d8-a10be0b2836b
gap = findmotifs2(
		load(Float64, "data/GAP.mp.dists.gz"), 
		load(Int, "data/GAP.mp.idx.gz"), 10, 600; top=actual["GAP"])

# ╔═╡ 12539f58-6689-4faf-b76b-4c655fefef25
freezer = findmotifs2(
		load(Float64, "data/freezer.mp.dists.gz"), 
		load(Int, "data/freezer.mp.idx.gz"), 10, 5000; top=actual["freezer"])

# ╔═╡ 344cec1b-9909-4c33-9d9c-081870680201
astro = findmotifs2(
		load(Float64, "data/astro.mp.dists.gz"), 
		load(Int, "data/astro.mp.idx.gz"), 10, 100; top=actual["ASTRO"])

# ╔═╡ 40d95c0f-dd33-48a2-a942-7241ef2d4a0b
ecg = findmotifs2(
		load(Float64, "data/ecg.mp.dists.gz"), 
		load(Int, "data/ecg.mp.idx.gz"), 10, 1000; top=actual["ECG"])

# ╔═╡ 88d0127d-f272-4b63-a631-d55a0f6c3929
seismic = findmotifs2(
		load(Float64, "data/VCAB_BP2_580_days-100000000.mp.dists.gz"), 
		load(Int, "data/VCAB_BP2_580_days-100000000.mp.idx.gz"), 10, 1000; top=actual["Seismic"])

# ╔═╡ 07a42c2b-e9ed-4704-99b7-6b51d0fe58e6
baseline = Dict(
	"ECG" => ecg,
	"ASTRO" => astro,
	"freezer" => freezer,
	"GAP" => gap,
	"HumanY" => humany,
	"Seismic" => seismic
)

# ╔═╡ 4e302e0f-b529-490d-b106-435d59cb6447
open("baselines.json", "w") do f
	JSON.print(f, baseline, 2)
end

# ╔═╡ 7d955800-929e-4f15-898f-f233b7e93ead
findmotifs2(dists, idx, 10, 5000; top=actual["freezer"])

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
# ╠═07a42c2b-e9ed-4704-99b7-6b51d0fe58e6
# ╠═52f5be68-d380-4fb2-8030-75db1186722f
# ╠═5dbfa9d3-a6a0-4be3-a3d8-a10be0b2836b
# ╠═12539f58-6689-4faf-b76b-4c655fefef25
# ╠═344cec1b-9909-4c33-9d9c-081870680201
# ╠═40d95c0f-dd33-48a2-a942-7241ef2d4a0b
# ╠═88d0127d-f272-4b63-a631-d55a0f6c3929
# ╠═7f9ec4f2-947f-11ec-05bc-df604fbd0243
# ╠═1d763f26-d916-4833-bad0-12624ce16072
# ╠═8845a7e9-288e-468a-9656-b938774ec08d
# ╠═4e302e0f-b529-490d-b106-435d59cb6447
# ╠═8f771188-5782-48f8-b894-fdba7096b599
# ╠═7d955800-929e-4f15-898f-f233b7e93ead
# ╠═27767505-36fc-49af-9d8c-77fadeee41fe
# ╠═1432d42f-feb5-43ef-8481-f69fd9ae9a8a
# ╠═865014a5-4183-4999-b6d4-83e01430a855
# ╠═71406a9f-2376-4c0a-b3d7-a20c841cceeb
# ╠═a777e6dc-31f3-4044-802a-01f30aa93e75
# ╠═07275457-b76b-43d8-9fa0-f0c813eec88a
# ╠═01d215d5-bc6f-4d89-8789-b4fb5704c90e
# ╠═3cea46f9-817c-4472-ab57-bb9e9be943b9
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
