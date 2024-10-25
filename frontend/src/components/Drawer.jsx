import React, { useState, useCallback } from 'react';
import { Grid, Typography } from '@mui/material';

import Drawer from '../ui/drawer/Drawer.js';
import CloseIcon from '@mui/icons-material/Close';
import Tappable from '../ui/tappable/Tappable';

export const AppDrawer = ({ children, open, close }) => {
    return (
      <Drawer
        anchor={"bottom"}
        open={open}
      > 
        <Grid position='relative'>
          <Grid 
            display='flex' justifyContent='center' alignItems='center'
            position='absolute'
            right='15px'
            top='-60px'
            onClick={close}
          >
            <Tappable
              style={{
                border: 'var(--element-border)',
                borderRadius: '50%',
                width: '46px',
                height: '46px',
                background: 'var(--bg-color)'
              }}
            >
              <CloseIcon sx={{ color: 'var(--hint-color)' }} />
            </Tappable>
          </Grid>
          <Grid borderRadius='var(--border-radius-lg) var(--border-radius-lg) 0 0' overflow={'hidden'} sx={{transform: 'translateZ(0)'}}>
            <Grid sx={{ overflowY: 'scroll', height: 'calc(100vh - 80px)' }}>
                {children}
            </Grid>
          </Grid>
        </Grid>
      </Drawer>
    );
  }
  
 export const useDrawer = () => {
    const [isOpen, setIsOpen] = useState(true);
  
    const toggle = useCallback(() => {
      setIsOpen((prev) => !prev);
    }, []);
  
    return [isOpen, toggle];
  }