require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'cunn'
require 'image'
require 'io'
package.path = package.path .. ";/home/vashishtm/ImageGen/neuraltalk2/?.lua;/home/vashishtm/ImageGen/neuraltalk2/model/?.t7"

--print(_VERSION)

util = paths.dofile('util_ntalk.lua')
ntalk_util = require('external')
--w2vutil = require 'w2vutils'
local ntalk_model =  '../neuraltalk2/model/karpathymodel.t7'
nngraph.setDebug(true)
opt = {
   dataset = 'lsun',       -- imagenet / lsun / folder
   batchSize = 64,
   loadSize = 96,
   fineSize = 64,
   nz = 300,               -- #  of dim for Z
   ncond = 300, 	   -- #  of dim for C
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 4,           -- #  of data loading threads to use
   niter = 100,            -- #  of iter at starting learning rate
   lr = 0.00015,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'experiment1',
   noise = 'normal',       -- uniform / normal
   conditional = true,
   checkpoint = 0,
   checkpoint_dir = 'checkpoints_ntalk_nobn/',
   plot_dir = 'plots_ntalk2_nobn/',
   im_dir = 'genimg_ntalk2_nobn'
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.DoubleTensor')

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
----------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local nc = 3
local nz = opt.nz
local ncond = opt.ncond
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0
local conditional = opt.conditional

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution
local JoinTable = nn.JoinTable

local nets = {}

function nets.generativeNet(ngf, noise_size, cond_size, conditional, out_size)
    -- local noise = nn.Identity()()
    -- local cond = nn.Identity()()
    
    -- input is Z + C, going into a convolution
    -- concatenation table
    -- local concat1 = JoinTable(2)({noise, cond})
    
    
    --concatenate before calling net.forward
    local noise = nn.Identity()()
    local cond = nn.Identity()()
    --local cond2 = nn.Identity()()
    --local cond3 = nn.Identity()()
    --local cond4 = nn.Identity()()
    --local cond5 = nn.Identity()()

    local data = JoinTable(2)({noise,cond})
    local data2 = SpatialFullConvolution(nz+ncond,100,1,1)(data)

    local data3 = SpatialBatchNormalization(100)(data2)

    local conv1 = SpatialFullConvolution(100, ngf * 8, 4, 4)(data3)
    local bn1 = SpatialBatchNormalization(ngf * 8)(conv1)
    local relu1 = nn.ReLU(true)(bn1)

    --local concat2 = JoinTable(2)({relu1,cond2})
    --local conv2 = (SpatialFullConvolution(ngf * 8 + ncond, ngf * 4, 4, 4, 2, 2, 1, 1))(concat2)
    local conv2 = (SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))(relu1)
    local bn2 = (SpatialBatchNormalization(ngf * 4))(conv2)
    local relu2 = nn.ReLU(true)(bn2)

    --local concat3 = JoinTable(2)({relu2,cond3})
    --local conv3 = SpatialFullConvolution(ngf * 4 + ncond, ngf * 2, 4, 4, 2, 2, 1, 1)(concat3)
    local conv3 = (SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))(relu2)
    local bn3 = SpatialBatchNormalization(ngf * 2)(conv3)
    local relu3 = nn.ReLU(true)(bn3)

    --local concat4 = JoinTable(2)({relu3,cond4})
    --local conv4 = SpatialFullConvolution(ngf * 2 + ncond, ngf, 4, 4, 2, 2, 1, 1)(concat4)
    local conv4 = SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1)(relu3)
    local bn4 = SpatialBatchNormalization(ngf)(conv4)
    local relu4 = nn.ReLU(true)(bn4)

    --local concat5 = JoinTable(2)({relu4,cond5})
    --local conv5 = SpatialFullConvolution(ngf + ncond, nc, 4, 4, 2, 2, 1, 1)(concat5)
    local conv5 = SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1)(relu4)
    local output = nn.Tanh()(conv5)

    --local netG = nn.gModule({noise,cond,cond2,cond3,cond4,cond5}, {output})
    local netG = nn.gModule({noise,cond},{output})
    return netG
end
function nets.discriminativeNet2(ndf)
    --local netD = nn.Sequential()
    -- Discriminative Network
	--local netD = nn.Sequential()
	local data = nn.Identity()()
	local cond = nn.Identity()()
	--local cond2 = nn.Identity()()
	--local cond3 = nn.Identity()()
	--local cond4 = nn.Identity()()
	--local cond5 = nn.Identity()()

	-- input is (nc) x 64 x 64
	--local concat1 = JoinTable(2)({data, cond})
	local conv1 = SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1)(data)
	local relu1 = nn.LeakyReLU(0.2, true)(conv1)
	-- state size: (ndf) x 32 x 32
	--local concat2 = JoinTable(2)({relu1, cond2})
	--local conv2 = SpatialConvolution(ndf+ncond, ndf * 2, 4, 4, 2, 2, 1, 1)(concat2)
	local conv2 = SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1)(relu1)
	local relu2 = nn.LeakyReLU(0.2, true)(conv2)
	-- state size: (ndf*2) x 16 x 16
	--local concat3 = JoinTable(2)({relu2, cond3})
	--local conv3 = SpatialConvolution(ndf * 2 + ncond, ndf * 4, 4, 4, 2, 2, 1, 1)(concat3)
	local conv3 = SpatialConvolution(ndf * 2 , ndf * 4, 4, 4, 2, 2, 1, 1)(relu2)
	local relu3 = nn.LeakyReLU(0.2, true)(conv3)
	-- state size: (ndf*4) x 8 x 8
	--local concat4 = JoinTable(2)({relu3, cond4})
	--local conv4 = SpatialConvolution(ndf * 4 + ncond, ndf * 8, 4, 4, 2, 2, 1, 1)(concat4)
	local conv4 = SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1)(relu3)
	local relu4 = nn.LeakyReLU(0.2, true)(conv4)
	-- state size: (ndf*8) x 4 x 4
	--local concat5 = JoinTable(2)({relu4,cond})
	--local conv5 = SpatialConvolution(ndf * 8 + ncond, 1, 4, 4)(concat5)
	--local conv5 = SpatialConvolution(ndf * 8, 1, 4, 4)(relu4)
	local reshape5 = nn.Reshape(ndf*8*4*4,1,1)(relu4)
	local concat5 = JoinTable(2)({reshape5,cond})
	local reshape52 = nn.Reshape(ndf*8*4*4 + ncond)(concat5)
	local linear5 = nn.Linear(ndf*8*4*4 + ncond,1)(reshape52)
	local sigmoid1 = nn.Sigmoid()(linear5)
	--local sigmoid1 = nn.Sigmoid()(conv5)
	-- state size: 1 x 1 x 1
	--local view1 = nn.View(1):setNumInputDims(3)(sigmoid1)
	--local netD = nn.gModule({data,cond,cond2,cond3,cond4,cond5},{view1})
	--local netD = nn.gModule({data,cond},{view1})
	local netD = nn.gModule({data,cond},{sigmoid1})
	return netD
end

if opt.checkpoint == 0 then
	netG = nets.generativeNet(ngf, noise_size, cond_size, conditional, out_size)
	netD = nets.discriminativeNet2(ndf)
	netG:apply(weights_init)
	netD:apply(weights_init)
else
	netG = util.load(opt.checkpoint_dir .. opt.name .. '_' .. opt.checkpoint .. '_net_G.t7', opt.gpu)
	netD = util.load(opt.checkpoint_dir .. opt.name .. '_' .. opt.checkpoint .. '_net_D.t7', opt.gpu)
end
local criterion = nn.BCECriterion()
---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local noise = torch.Tensor(opt.batchSize, nz, 1, 1)
local cond = torch.Tensor(opt.batchSize, ncond, 1, 1)
--local cond2 = torch.Tensor(opt.batchSize, ncond, 4, 4)
--local cond3 = torch.Tensor(opt.batchSize, ncond, 8, 8)
--local cond4 = torch.Tensor(opt.batchSize, ncond, 16, 16)
--local cond5 = torch.Tensor(opt.batchSize, ncond, 32, 32)
local label = torch.Tensor(opt.batchSize)
local errD, errG, errN, nGrad
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------
if opt.gpu > 0 then
   print('yo!')
   cutorch.setDevice(opt.gpu)
   input = input:cuda();  noise = noise:cuda();  label = label:cuda(); cond = cond:cuda();
   netG = util.cudnn(netG);     netD = util.cudnn(netD)
   netD:cuda();           netG:cuda();           criterion:cuda()
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

if opt.display then disp = require 'display' end

noise_vis = noise:clone()

if opt.noise == 'uniform' then
    noise_vis:uniform(-1, 1)
elseif opt.noise == 'normal' then
    noise_vis:normal(0, 1)
end
-- load neuraltalk networks
local usentalk = false
if usentalk then
	cutorch.setDevice(3)
	ntalk_protos = ntalk_util.getProtos(ntalk_model, 3)
	cutorch.setDevice(opt.gpu)
end
local counter = 0
epoch = 1
local fDxCond = function(x)

   netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

   gradParametersD:zero()

   -- train with real
   data_tm:reset(); data_tm:resume()
   local real, captions, scalarLabels, vocab_vecs = data:getBatch()
   data_tm:stop()
   real = real:cuda()
   captions = captions:cuda()

   input:copy(real)
   label:fill(real_label)
   cond:copy(captions) 

   fuck = {input, cond}
   print(fuck)
   netD:cuda()
   local output = netD:forward(fuck) -- run with image + conditional info

   local errD_real = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward({input, cond}, df_do)

   -- train with fake
   if opt.noise == 'uniform' then -- regenerate random noise
       noise:uniform(-1, 1)
   elseif opt.noise == 'normal' then
       noise:normal(0, 1)
   end

   local fake = netG:forward({noise,cond})
   input:copy(fake)
   label:fill(fake_label)

   -- run discriminative net on generated image
   local output = netD:forward({input, cond})
   local errD_fake = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward({input, cond}, df_do)


   errD = errD_real + errD_fake
   if usentalk then
   	cutorch.setDevice(3)
   	errN, nGrad = ntalk_util.getLoss(ntalk_protos, fake, vocab_vecs)
   	cutorch.setDevice(opt.gpu)
   end
   return errD, gradParametersD
end

local fGxCond = function(x)
   netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

   gradParametersG:zero()

   --[[ the three lines below were already executed in fDx, so save computation
   noise:uniform(-1, 1) -- regenerate random noise
   local fake = netG:forward(noise)
   input:copy(fake) ]]--
   label:fill(real_label) -- fake labels are real for generator cost

   local output = netD.output -- netD:forward(input) was already executed in fDx, so save computation
   errG = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   local df_dg = netD:updateGradInput({input, cond}, df_do)
   local alpha = 1
   local beta = 1
   if counter % 50 == 0 then
   	image.save(opt.im_dir .. '/gengrad' .. epoch .. '_' .. counter ..'.png', image.toDisplayTensor(df_dg * 50))
   	image.save(opt.im_dir .. '/img' .. epoch .. '_' .. counter .. '.png', image.toDisplayTensor(input))
   	--image.save(opt.im_dir .. '/genntalk' .. epoch .. '_' .. counter .. '.png', image.toDisplayTensor(nGrad * 50))
   end
   if usentalk then
      	nGrad = nGrad * beta
   	df_di = df_dg + nGrad:cuda()
   else
	df_di = df_dg[1]
   end
   netG:backward({noise,cond},df_di)
   print(errN)
   if usentalk then
   	return alpha * errG + beta * .005 * errN, gradParametersG
   else
	return errG, gradParametersG
   end
end
local errDTable = {}
local errGTable = {}
local ntalkTable = {}

saveTable = function(t, filename)
    local f = io.open(filename, 'w')
    for i=1,#t do
	f:write(t[i])
	f:write(',')
    end
    f:write('\n')
    f:close()
end
-- train
for epoch = 1, opt.niter do
   epoch_tm:reset()
   counter = 0
   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      tm:reset()
      -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      optim.adam(fDxCond, parametersD, optimStateD)
      -- (2) Update G network: maximize log(D(G(z)))
      optim.adam(fGxCond, parametersG, optimStateG)

      -- display
      counter = counter + 1
      if counter % 10 == 0 and opt.display then
          --local data = torch.cat(noise_vis, cond, 2)
          --local fake = netG:forward(data)
          --local real = data:getBatch()
          --disp.image(fake, {win=opt.display_id, title=opt.name})
          --disp.image(real, {win=opt.display_id * 3, title=opt.name})
      end

      -- logging
      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Err_G: %.4f  Err_D: %.4f'):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, data_tm:time().real,
                 errG and errG or -1, errD and errD or -1))
	table.insert(errDTable, errD and errD or -1)
	table.insert(errGTable, errG and errG or -1)
	table.insert(ntalkTable, errN and errN or -1)
      end
   end
   paths.mkdir(opt.checkpoint_dir)
   parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
   parametersG, gradParametersG = nil, nil
   print(netG)
   if epoch % 10 == 0 then
	real_epoch = opt.checkpoint + epoch
   	util.save(opt.checkpoint_dir .. opt.name .. '_' .. real_epoch .. '_net_G.t7', netG, opt.gpu)
   	util.save(opt.checkpoint_dir .. opt.name .. '_' .. real_epoch .. '_net_D.t7', netD, opt.gpu)
	saveTable(errDTable, opt.plot_dir .. 'errD_' .. opt.name .. '_' .. real_epoch .. '.txt')
	saveTable(errGTable, opt.plot_dir .. 'errG_' .. opt.name .. '_' .. real_epoch .. '.txt')
	saveTable(ntalkTable, opt.plot_dir .. 'ntalkLoss_' .. opt.name .. '_' .. real_epoch .. '.txt')
   end
   parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
   parametersG, gradParametersG = netG:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end
