-- script to preprocess w2v representations of all the training captions
local w2vutil = require 'w2vutils'
local vigo = require 'csvigo'


local caption_dir = "/home/vashishtm/ImageGen/captionsSmall3/"
local output_dir = "/home/vashishtm/ImageGen/captionsSmallVec3/"
local f = io.popen('ls -a "'..caption_dir..'"')
local t = {}
local i = 0
for fname in f:lines() do
	i = i+1
	t[i] = fname
end

--print "T"
--print(t)

f:close()
local caption_rep
local cap_vec
local caption
table.remove(t,1)
table.remove(t,1)


for k,v in pairs(t) do
	print(v)
	caption_rep = {}
	local c_path = v:gsub('txt','csv')
	local cap_file = io.open(caption_dir .. v)
	caption = cap_file:read("*a")
	print(caption)
	cap_file:close()
	cap_vec = torch.zeros(300)
       	for word in caption:gmatch("%w+") do
		cap_vec = cap_vec + w2vutil:word2vec(word)
	end
	table.insert(caption_rep,cap_vec)
	print(caption_rep)
	--vigo.save({path= output_dir .. c_path, data=caption_rep})
	torch.save(output_dir .. c_path, cap_vec)
end


