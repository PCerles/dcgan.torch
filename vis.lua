require 'image'
require 'nn'
require 'nngraph'
require 'cunn'
--vigo = require 'csvigo'
util = paths.dofile('util.lua')
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
    net = '',              -- path to the generator network
    layer= 2
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
net = util.load(opt.net, opt.gpu)
print(net:get(opt.layer).weight)
image.toDisplayTensor(net:get(opt.layer).weight)
