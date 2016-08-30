"""
This iterator will return tensors of dimension (8, 16, 1152, 864),
each tensor corresponding to one of the 365 days of the year
"""
def _day_iterator(year=1979):
    variables = [u'PRECT',
                 u'PS',
                 u'PSL',
                 u'QREFHT',
                 u'T200',
                 u'T500',
                 u'TMQ',
                 u'TREFHT',
                 u'TS',
                 u'U850',
                 u'UBOT',
                 u'V850',
                 u'VBOT',
                 u'Z1000',
                 u'Z200',
                 u'ZBOT']    
    # this directory can be accessed from cori
    maindir="/project/projectdirs/dasrepo/gordon_bell/climate/data/big_images/" + \
    str(year) 
    lsdir=listdir(maindir)
    rpfile = re.compile(r"^cam5_.*\.nc$")
    camfiles = [f for f in lsdir if rpfile.match(f)]
    for camfile in camfiles:
        datasets = [ nc.Dataset(maindir+'/'+camfile, "r", format="NETCDF4") ]
        time_steps=8
        x=768
        y=1152
        day_slices = [make_spatiotemporal_tensor(dataset,time_steps,variables) for dataset in datasets]
        tr_data = np.vstack(day_slices).reshape(len(datasets), time_steps,len(variables), x, y)
        yield tr_data[0]

