require 'image'
require 'nn'
require 'nngraph'
require 'cunn'
w2vutil = require 'w2vutils'
--vigo = require 'csvigo'
util = paths.dofile('util.lua')
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
    batchSize = 64,        -- number of samples to produce
    noisetype = 'normal',  -- type of noise distribution (uniform / normal).
    net = '',              -- path to the generator network
    imsize = 1,            -- used to produce larger images. 1 = 64px. 2 = 80px, 3 = 96px, ...
    noisemode = 'random',  -- random / line / linefull1d / linefull
    name = 'gen_img/',  -- name of the file saved
    gpu = 2,               -- gpu mode. 0 = CPU, 1 = GPU
    display = 1,           -- Display image: 0 = false, 1 = true
    nz = 300,
    ncond = 300,              
    input_caption = 'fruit',
    own_caption = 1
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end
assert(net ~= '', 'provide a generator model')
noise = torch.Tensor(opt.batchSize, opt.nz, opt.imsize, opt.imsize)
if opt.own_caption==1 then
	local caption = opt.input_caption
	--w2vutil = require 'w2vutils'
	temp_rep = torch.zeros(1, 300)
	for word in caption:gmatch("%w+") do
		temp_rep = temp_rep + w2vutil:word2vec(word)
	end
	caption_rep = torch.reshape(temp_rep, opt.ncond, 1)
	caption_rep = torch.expand(caption_rep,opt.ncond,opt.batchSize):transpose(1,2)
	caption_rep = torch.reshape(caption_rep,opt.batchSize,opt.ncond,1,1)
else
	print(opt.input_caption)
	print(io.open("/home/vashishtm/ImageGen/captionsSmall/" .. opt.input_caption .. ".txt"):read())
	caption_path = "/home/vashishtm/ImageGen/captionsSmallVec/" .. opt.input_caption .. ".csv"
	caption_rep = torch.load(caption_path)
	caption_rep = torch.reshape(caption_rep, opt.ncond, 1)
	caption_rep = torch.expand(caption_rep,opt.ncond,opt.batchSize):transpose(1,2)
	--caption_rep = torch.expand(caption_rep, opt.batchSize, opt.ncond,1,1) 
	caption_rep = torch.reshape(caption_rep,opt.batchSize,opt.ncond,1,1)
end
net = util.load(opt.net, opt.gpu)
-- for older models, there was nn.View on the top
-- which is unnecessary, and hinders convolutional generations.
if torch.type(net:get(1)) == 'nn.View' then
    net:remove(1)
end

if opt.noisetype == 'uniform' then
    noise:uniform(-1, 1)
elseif opt.noisetype == 'normal' then
    noise:normal(0, 1)
end

noiseL = torch.FloatTensor(opt.nz):uniform(-1, 1)
noiseR = torch.FloatTensor(opt.nz):uniform(-1, 1)
if opt.noisemode == 'line' then
   -- do a linear interpolation in Z space between point A and point B
   -- each sample in the mini-batch is a point on the line
    line  = torch.linspace(0, 1, opt.batchSize)
    for i = 1, opt.batchSize do
        noise:select(1, i):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    end
elseif opt.noisemode == 'linefull1d' then
   -- do a linear interpolation in Z space between point A and point B
   -- however, generate the samples convolutionally, so a giant image is produced
    assert(opt.batchSize == 1, 'for linefull1d mode, give batchSize(1) and imsize > 1')
    noise = noise:narrow(3, 1, 1):clone()
    line  = torch.linspace(0, 1, opt.imsize)
    for i = 1, opt.imsize do
        noise:narrow(4, i, 1):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    end
elseif opt.noisemode == 'linefull' then
   -- just like linefull1d above, but try to do it in 2D
    assert(opt.batchSize == 1, 'for linefull mode, give batchSize(1) and imsize > 1')
    line  = torch.linspace(0, 1, opt.imsize)
    for i = 1, opt.imsize do
        noise:narrow(3, i, 1):narrow(4, i, 1):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    end
end
if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
    net:cuda()
    util.cudnn(net)
    noise = noise:cuda()
else
   net:float()
end

-- a function to setup double-buffering across the network.
-- this drastically reduces the memory needed to generate samples
util.optimizeInferenceMemory(net)
--local data = torch.cat(noise, caption_rep, 2)
--local cond2 = torch.expand(caption_rep,opt.batchSize,opt.ncond,4,4)
--local cond3 = torch.expand(caption_rep,opt.batchSize,opt.ncond,8,8)
--local cond4 = torch.expand(caption_rep,opt.batchSize,opt.ncond,16,16)
--local cond5 = torch.expand(caption_rep,opt.batchSize,opt.ncond,32,32)
paths.mkdir(opt.name .. '/' .. opt.net)
local images = net:forward({noise,caption_rep})
print('Images size: ', images:size(1)..' x '..images:size(2) ..' x '..images:size(3)..' x '..images:size(4))
images:add(1):mul(0.5)
print('Min, Max, Mean, Stdv', images:min(), images:max(), images:mean(), images:std())
image.save(opt.name .. opt.net .. '/' .. opt.input_caption .. '.png', image.toDisplayTensor(images))
print('Saved image to: ', opt.name .. opt.net .. '/' ..  opt.input_caption .. '.png')

--for i=1,images:size(1) do
--    image.save(opt.name .. opt.net ..'/' .. opt.input_caption .. '_' .. i .. '.png', image.toDisplayTensor(images[i]))
--    print('Saved image ' .. i)
--end

if opt.display then
    disp = require 'display'
    disp.image(images)
    print('Displayed image')
end
